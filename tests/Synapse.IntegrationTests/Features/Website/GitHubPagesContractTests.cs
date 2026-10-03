namespace ManagedCode.Synapse.IntegrationTests.Features.Website;

public sealed class GitHubPagesContractTests
{
    [Test]
    public async Task PagesWorkflowPublishesCanonicalContent()
    {
        var root = FindRepositoryRoot();
        var workflow = await File.ReadAllTextAsync(
            Path.Combine(root, ".github", "workflows", "pages.yml"));

        await Assert.That(workflow).Contains("docs/images/**");
        await Assert.That(workflow).Contains("site/**");
        await Assert.That(workflow).Contains("workflow_dispatch:");
        await Assert.That(workflow).Contains("pages: write");
        await Assert.That(workflow).Contains("id-token: write");
        await Assert.That(workflow).Contains("name: github-pages");
        await Assert.That(workflow).Contains("actions/configure-pages@");
        await Assert.That(workflow).Contains("actions/upload-pages-artifact@");
        await Assert.That(workflow).Contains("actions/deploy-pages@");
        await Assert.That(workflow).Contains(
            "actions/configure-pages@983d7736d9b0ae728b81ab479565c72886d7745b");
        await Assert.That(workflow).Contains(
            "actions/upload-pages-artifact@7b1f4a764d45c48632c6b24a0339c27f5614fb0b");
        await Assert.That(workflow).Contains(
            "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e");
        await Assert.That(workflow).Contains(
            "cp site/index.html site/styles.css site/site.js site/favicon.svg _site/");
        await Assert.That(workflow).Contains("cp docs/images/*.svg _site/images/");
        await Assert.That(workflow).Contains("cp site/metrics.js site/comparisons.js site/comparisons.css _site/");
        await Assert.That(workflow).Contains("path: ./_site");

        var index = await File.ReadAllTextAsync(Path.Combine(root, "site", "index.html"));
        await Assert.That(index).Contains("href=\"comparisons.css\"");
        foreach (var image in ExtractLocalImages(index))
        {
            await Assert.That(File.Exists(Path.Combine(root, "docs", image))).IsTrue()
                .Because($"Site image '{image}' must be published from a real file.");
        }
    }

    private static IEnumerable<string> ExtractLocalImages(string html)
    {
        const string Marker = "src=\"images/";
        var start = 0;
        while ((start = html.IndexOf(Marker, start, StringComparison.Ordinal)) >= 0)
        {
            var targetStart = start + "src=\"".Length;
            var targetEnd = html.IndexOf('"', targetStart);
            if (targetEnd < 0)
            {
                yield break;
            }

            yield return html[targetStart..targetEnd];
            start = targetEnd + 1;
        }
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Synapse.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the Synapse repository root.");
    }
}
