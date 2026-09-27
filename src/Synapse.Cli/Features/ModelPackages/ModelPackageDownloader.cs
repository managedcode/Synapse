using System.Buffers;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.Catalog;

namespace ManagedCode.Synapse.Cli.Features.ModelPackages;

internal sealed class ModelPackageDownloader(HttpClient client)
{
    private const int BufferSize = 1024 * 1024;
    private const int MaximumRedirects = 5;

    public async Task<DownloadedModelPackage> FetchAsync(
        ModelPackageDefinition package,
        string outputRoot,
        CancellationToken cancellationToken)
    {
        var outputRootFull = Path.GetFullPath(outputRoot);
        var packageRoot = Path.GetFullPath(Path.Combine(outputRootFull, package.Id));
        EnsureChildPath(outputRootFull, packageRoot, "Model package resolves outside the output directory.");
        _ = Directory.CreateDirectory(packageRoot);
        var downloaded = 0;
        var reused = 0;
        foreach (var file in package.Files)
        {
            var destination = ResolveDestination(packageRoot, file.Path);
            if (await HasExpectedContentAsync(destination, file, cancellationToken).ConfigureAwait(false))
            {
                reused++;
                continue;
            }

            await DownloadAsync(package, file, destination, cancellationToken).ConfigureAwait(false);
            downloaded++;
        }

        return new DownloadedModelPackage(package.Id, packageRoot, downloaded, reused);
    }

    private async Task DownloadAsync(
        ModelPackageDefinition package,
        ModelPackageFile file,
        string destination,
        CancellationToken cancellationToken)
    {
        var directory = Path.GetDirectoryName(destination)
            ?? throw new InvalidOperationException("Model destination has no parent directory.");
        _ = Directory.CreateDirectory(directory);
        var temporary = destination + $".partial-{Guid.NewGuid():N}";
        try
        {
            using var response = await SendWithRedirectsAsync(package, file.Url, cancellationToken)
                .ConfigureAwait(false);
            _ = response.EnsureSuccessStatusCode();
            if (response.Content.Headers.ContentLength is { } contentLength && contentLength != file.SizeBytes)
            {
                throw new InvalidDataException(
                    $"Downloaded '{file.Path}' declared {contentLength} bytes; expected {file.SizeBytes}.");
            }

            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            var (length, digest) = await CopyPayloadAsync(
                source,
                temporary,
                file,
                cancellationToken).ConfigureAwait(false);
            if (length != file.SizeBytes || !digest.Equals(file.Sha256, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Downloaded '{file.Path}' failed size or SHA-256 verification.");
            }

            File.Move(temporary, destination, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary))
            {
                File.Delete(temporary);
            }
        }
    }

    internal static async Task<(long Length, string Digest)> CopyPayloadAsync(
        Stream source,
        string temporary,
        ModelPackageFile file,
        CancellationToken cancellationToken)
    {
        await using var target = new FileStream(
            temporary,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            BufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        long length = 0;
        try
        {
            int read;
            while ((read = await source.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false)) > 0)
            {
                var newLength = checked(length + read);
                if (newLength > file.SizeBytes)
                {
                    throw new InvalidDataException(
                        $"Downloaded '{file.Path}' exceeded its expected size of {file.SizeBytes} bytes.");
                }

                await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                hash.AppendData(buffer, 0, read);
                length = newLength;
            }

            await target.FlushAsync(cancellationToken).ConfigureAwait(false);
            return (length, Convert.ToHexStringLower(hash.GetHashAndReset()));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private async Task<HttpResponseMessage> SendWithRedirectsAsync(
        ModelPackageDefinition package,
        Uri initialUri,
        CancellationToken cancellationToken)
    {
        var sourceHost = new Uri(package.SourceUrl, UriKind.Absolute).Host;
        var current = initialUri;
        for (var redirectCount = 0; ; redirectCount++)
        {
            EnsureTrustedUri(current, sourceHost, redirectCount == 0);
            using var request = new HttpRequestMessage(HttpMethod.Get, current);
            request.Headers.AcceptEncoding.Add(new StringWithQualityHeaderValue("identity"));
            var response = await client.SendAsync(
                request,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken).ConfigureAwait(false);
            if (!IsRedirect(response.StatusCode))
            {
                return response;
            }

            if (redirectCount >= MaximumRedirects || response.Headers.Location is null)
            {
                response.Dispose();
                throw new InvalidDataException($"Model download exceeded {MaximumRedirects} redirects.");
            }

            current = response.Headers.Location.IsAbsoluteUri
                ? response.Headers.Location
                : new Uri(current, response.Headers.Location);
            response.Dispose();
        }
    }

    private static bool IsRedirect(HttpStatusCode statusCode) => statusCode is
        HttpStatusCode.MovedPermanently or
        HttpStatusCode.Redirect or
        HttpStatusCode.RedirectMethod or
        HttpStatusCode.TemporaryRedirect or
        HttpStatusCode.PermanentRedirect;

    internal static void EnsureTrustedUri(Uri uri, string sourceHost, bool isInitial)
    {
        var trustedHost = uri.Host.Equals(sourceHost, StringComparison.OrdinalIgnoreCase) ||
            (!isInitial && (IsHostOrSubdomain(uri.Host, "huggingface.co") ||
                            IsHostOrSubdomain(uri.Host, "hf.co")));
        if (uri.Scheme != Uri.UriSchemeHttps || !trustedHost || !uri.IsDefaultPort)
        {
            throw new InvalidDataException($"Model download URI '{uri}' is not trusted.");
        }
    }

    private static bool IsHostOrSubdomain(string host, string suffix) =>
        host.Equals(suffix, StringComparison.OrdinalIgnoreCase) ||
        host.EndsWith('.' + suffix, StringComparison.OrdinalIgnoreCase);

    private static async Task<bool> HasExpectedContentAsync(
        string destination,
        ModelPackageFile file,
        CancellationToken cancellationToken)
    {
        var information = new FileInfo(destination);
        if (!information.Exists || information.Length != file.SizeBytes)
        {
            return false;
        }

        await using var stream = new FileStream(
            destination,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            BufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        var digest = Convert.ToHexStringLower(
            await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false));
        return digest.Equals(file.Sha256, StringComparison.Ordinal);
    }

    private static string ResolveDestination(string packageRoot, string relativePath)
    {
        var destination = Path.GetFullPath(Path.Combine(packageRoot, relativePath));
        EnsureChildPath(packageRoot, destination, "Model file resolves outside its package directory.");
        return destination;
    }

    private static void EnsureChildPath(string root, string candidate, string message)
    {
        var requiredPrefix = root.EndsWith(Path.DirectorySeparatorChar)
            ? root
            : root + Path.DirectorySeparatorChar;
        var comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;
        if (!candidate.StartsWith(requiredPrefix, comparison))
        {
            throw new InvalidDataException(message);
        }
    }
}

internal sealed record DownloadedModelPackage(
    string Id,
    string Directory,
    int DownloadedFiles,
    int ReusedFiles);
