namespace ManagedCode.Synapse.Runtime.Features.ModelPackages;

/// <summary>Read-only logical stream over a bounded slice, without owning the underlying handle.</summary>
internal sealed class CompiledReadStream(Stream source, long start, long length) : Stream
{
    private long _position;

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => length;
    public override long Position { get => _position; set => Seek(value, SeekOrigin.Begin); }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int Read(Span<byte> buffer)
    {
        source.Position = checked(start + _position);
        var read = source.Read(buffer[..(int)Math.Min(buffer.Length, length - _position)]);
        _position += read;
        return read;
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        var position = checked(offset + (origin switch
        {
            SeekOrigin.Begin => 0,
            SeekOrigin.Current => _position,
            SeekOrigin.End => length,
            _ => throw new ArgumentOutOfRangeException(nameof(origin)),
        }));
        if (position < 0 || position > length)
        {
            throw new InvalidDataException("Compiled package header seek is out of bounds.");
        }

        return _position = position;
    }

    public override void Flush() => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
