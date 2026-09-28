using System.Text;
using System.Text.Json;

internal static class LockedChatScenario
{
    public static List<DialogueTurn> Read(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        if (root.GetProperty("schemaVersion").GetInt32() != 1 ||
            root.GetProperty("kind").GetString() != "locked-chat")
        {
            throw new InvalidDataException("Unsupported locked-dialogue scenario.");
        }

        var system = root.GetProperty("system").GetString();
        var turns = root.GetProperty("turns").EnumerateArray().ToArray();
        if (string.IsNullOrWhiteSpace(system) || turns.Length is not (1 or 3))
        {
            throw new InvalidDataException("The diagnostic requires one or three nonempty turns.");
        }

        var prompt = new StringBuilder("System: ").Append(system).AppendLine();
        var result = new List<DialogueTurn>(turns.Length);
        foreach (var (turn, index) in turns.Select((turn, index) => (turn, index)))
        {
            var user = turn.GetProperty("user").GetString();
            var locked = turn.GetProperty("lockedAssistant").GetString();
            if (string.IsNullOrWhiteSpace(user) || string.IsNullOrWhiteSpace(locked))
            {
                throw new InvalidDataException("Dialogue turn and locked assistant text must be nonempty.");
            }

            _ = prompt.Append("User: ").Append(user).AppendLine().Append("Assistant:");
            result.Add(new DialogueTurn(index + 1, system, user, locked, prompt.ToString(), []));
            _ = prompt.Append(' ').Append(locked).AppendLine();
        }

        return result;
    }
}

internal sealed record DialogueTurn(int Number, string System, string User, string LockedAssistant,
    string Prompt, int[] PromptTokenIds);
