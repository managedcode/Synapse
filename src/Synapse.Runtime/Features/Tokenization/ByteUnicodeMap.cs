using System.Text;

namespace ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// GPT-2's reversible byte-to-character table: printable Latin-1 bytes map to themselves and the other 68 bytes
/// map to U+0100 onward, so every byte sequence becomes a string of visible vocabulary characters.
/// </summary>
internal static class ByteUnicodeMap
{
    private static readonly char[] ByteToChar = BuildTable();
    private static readonly Dictionary<char, byte> CharToByte = BuildInverse();

    public static string Encode(ReadOnlySpan<char> text)
    {
        var byteCount = Encoding.UTF8.GetByteCount(text);
        var bytes = new byte[byteCount];
        _ = Encoding.UTF8.GetBytes(text, bytes);
        return string.Create(bytes.Length, bytes, static (destination, source) =>
        {
            for (var index = 0; index < source.Length; index++)
            {
                destination[index] = ByteToChar[source[index]];
            }
        });
    }

    public static byte ToByte(char character) =>
        CharToByte.TryGetValue(character, out var value)
            ? value
            : throw new InvalidDataException($"Character U+{(int)character:X4} is not a byte-level BPE symbol.");

    private static char[] BuildTable()
    {
        var table = new char[256];
        var next = 256;
        for (var value = 0; value < 256; value++)
        {
            var printable = value is (>= '!' and <= '~') or (>= 0xA1 and <= 0xAC) or (>= 0xAE and <= 0xFF);
            table[value] = printable ? (char)value : (char)next++;
        }

        return table;
    }

    private static Dictionary<char, byte> BuildInverse()
    {
        var inverse = new Dictionary<char, byte>(256);
        for (var value = 0; value < 256; value++)
        {
            inverse[ByteToChar[value]] = (byte)value;
        }

        return inverse;
    }
}
