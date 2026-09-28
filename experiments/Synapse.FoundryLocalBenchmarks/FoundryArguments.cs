using System.Globalization;

internal sealed class FoundryArguments
{
    private readonly Dictionary<string, string> _values;

    private FoundryArguments(Dictionary<string, string> values)
    {
        _values = values;
    }

    public static FoundryArguments? Parse(string[] args)
    {
        if (args.Length % 2 != 0)
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Length; index += 2)
        {
            if (!args[index].StartsWith("--", StringComparison.Ordinal) ||
                !values.TryAdd(args[index], args[index + 1]))
            {
                return null;
            }
        }

        return new FoundryArguments(values);
    }

    public void AllowOnly(params string[] names)
    {
        var unknown = _values.Keys.FirstOrDefault(key => !names.Contains(key, StringComparer.Ordinal));
        if (unknown is not null)
        {
            throw new FoundryUsageException($"Unknown option '{unknown}'.");
        }
    }

    public string? Optional(string name) => _values.GetValueOrDefault(name);

    public string Require(string name) => _values.TryGetValue(name, out var value) &&
        !string.IsNullOrWhiteSpace(value)
            ? value
            : throw new FoundryUsageException($"Missing required option {name}.");

    public string Device()
    {
        var device = _values.GetValueOrDefault("--device", "cpu");
        return device is "cpu" or "gpu"
            ? device
            : throw new FoundryUsageException("--device must be cpu or gpu.");
    }

    public int Count(string name, int fallback, int minimum, int maximum)
    {
        if (!_values.TryGetValue(name, out var text))
        {
            return fallback;
        }

        return int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value) &&
            value >= minimum && value <= maximum
                ? value
                : throw new FoundryUsageException($"{name} must be an integer in [{minimum}, {maximum}].");
    }
}
