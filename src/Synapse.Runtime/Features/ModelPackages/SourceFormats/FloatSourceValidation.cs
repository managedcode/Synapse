using System.Numerics;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

internal static class FloatSourceValidation
{
    internal static int FirstNonFinite32(ReadOnlySpan<uint> words)
    {
        const uint ExponentMask = 0x7F80_0000;
        var index = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var mask = new Vector<uint>(ExponentMask);
            for (; index <= words.Length - Vector<uint>.Count; index += Vector<uint>.Count)
            {
                if (Vector.EqualsAny(new Vector<uint>(words[index..]) & mask, mask))
                {
                    return Scan32(words, index, index + Vector<uint>.Count);
                }
            }
        }

        return Scan32(words, index, words.Length);
    }

    internal static int FirstNonFinite16(ReadOnlySpan<ushort> words, ushort exponentMask)
    {
        var index = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var mask = new Vector<ushort>(exponentMask);
            for (; index <= words.Length - Vector<ushort>.Count; index += Vector<ushort>.Count)
            {
                if (Vector.EqualsAny(new Vector<ushort>(words[index..]) & mask, mask))
                {
                    return Scan16(words, exponentMask, index, index + Vector<ushort>.Count);
                }
            }
        }

        return Scan16(words, exponentMask, index, words.Length);
    }

    private static int Scan32(ReadOnlySpan<uint> words, int start, int end)
    {
        for (var index = start; index < end; index++)
        {
            if ((words[index] & 0x7F80_0000) == 0x7F80_0000)
            {
                return index;
            }
        }

        return -1;
    }

    private static int Scan16(ReadOnlySpan<ushort> words, ushort exponentMask, int start, int end)
    {
        for (var index = start; index < end; index++)
        {
            if ((words[index] & exponentMask) == exponentMask)
            {
                return index;
            }
        }

        return -1;
    }
}
