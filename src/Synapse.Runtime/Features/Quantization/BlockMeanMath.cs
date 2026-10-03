using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

internal static class BlockMeanMath
{
    public static void SumGroups(ReadOnlySpan<float> input, int groupElements, Span<double> sums)
    {
        for (var group = 0; group < sums.Length; group++)
        {
            var start = group * groupElements;
            var count = Math.Min(groupElements, input.Length - start);
            var sum = 0.0;
            foreach (var value in input.Slice(start, count))
            {
                sum += value;
            }

            sums[group] = sum;
        }
    }

    public static double Dot(ReadOnlySpan<byte> encodedMeans, ReadOnlySpan<double> sums)
    {
        var index = 0;
        var result = 0.0;
        if (BitConverter.IsLittleEndian && Vector.IsHardwareAccelerated)
        {
            var means = MemoryMarshal.Cast<byte, float>(encodedMeans);
            var lowSum = Vector<double>.Zero;
            var highSum = Vector<double>.Zero;
            for (; index <= sums.Length - Vector<float>.Count; index += Vector<float>.Count)
            {
                Vector.Widen(new Vector<float>(means[index..]), out var low, out var high);
                lowSum += low * new Vector<double>(sums[index..]);
                highSum += high * new Vector<double>(sums[(index + Vector<double>.Count)..]);
            }

            result = Vector.Sum(lowSum) + Vector.Sum(highSum);
        }

        for (; index < sums.Length; index++)
        {
            result += (double)BinaryPrimitives.ReadSingleLittleEndian(encodedMeans[(index * sizeof(float))..]) * sums[index];
        }

        return result;
    }
}
