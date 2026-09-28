using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// Exact FP16-to-FP32 conversion for finite values: shift the magnitude into FP32 position and rescale by
/// 2^112, which also produces exact subnormals and zero. Q8_0 scales are finite by construction.
/// </summary>
internal static class Fp16
{
    private const uint MagnitudeMask = 0x7FFF;
    private const uint SignMask = 0x8000;
    private const int RescaleBits = 0x7780_0000;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToSingle(ushort bits)
    {
        var magnitude = BitConverter.Int32BitsToSingle((int)((bits & MagnitudeMask) << 13)) *
            BitConverter.Int32BitsToSingle(RescaleBits);
        return (bits & SignMask) == 0 ? magnitude : -magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> ToSingle(ushort first, ushort second, ushort third, ushort fourth)
    {
        var bits = Vector128.Create(first, second, third, (uint)fourth);
        var magnitude = ((bits & Vector128.Create(MagnitudeMask)) << 13).AsSingle() *
            Vector128.Create(RescaleBits).AsSingle();
        return (magnitude.AsUInt32() | ((bits & Vector128.Create(SignMask)) << 16)).AsSingle();
    }
}
