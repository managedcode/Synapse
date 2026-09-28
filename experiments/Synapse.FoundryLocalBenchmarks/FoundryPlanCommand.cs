using System.Globalization;
using System.Text;
using System.Text.Json;

internal static class FoundryPlanCommand
{
    public static async Task<int> RunAsync(FoundryArguments arguments)
    {
        arguments.AllowOnly("--set", "--summary");
        var set = FoundryModelSet.Load(arguments.Require("--set"));
        var entries = new List<FoundryPlanEntry>();
        var coverage = new StringBuilder()
            .AppendLine("## Foundry Local plan — one isolated job per runner and model")
            .AppendLine()
            .Append("Set `").Append(set.Id)
            .AppendLine("`. A model is scheduled when its CPU variant file is at most half of the runner's memory.")
            .AppendLine()
            .AppendLine("| Runner | Model | Status | Family | CPU variant | File MB |")
            .AppendLine("|---|---|---|---|---|---:|");
        foreach (var runner in set.Runners)
        {
            foreach (var model in set.Models)
            {
                var cpu = model.Variant("cpu");
                var fits = FoundryModelSet.Fits(cpu, runner);
                if (fits)
                {
                    entries.Add(new FoundryPlanEntry(runner.Id, runner.Name, model.Family, model.Alias,
                        cpu.Id, cpu.FileSizeMb, runner.MemoryMb));
                }

                _ = coverage.Append('|').Append(runner.Id).Append('|').Append(model.Alias).Append('|')
                    .Append(fits ? "scheduled" : string.Create(CultureInfo.InvariantCulture,
                        $"excluded: {cpu.FileSizeMb} MB > half of {runner.MemoryMb} MB"))
                    .Append('|').Append(model.Family).Append("|`").Append(cpu.Id).Append("`|")
                    .Append(cpu.FileSizeMb.ToString(CultureInfo.InvariantCulture)).AppendLine("|");
            }
        }

        Console.WriteLine(JsonSerializer.Serialize(new FoundryMatrix(entries),
            FoundryJsonContext.Default.FoundryMatrix));
        Console.Error.Write(coverage.ToString());
        if (arguments.Optional("--summary") is { } summary)
        {
            await File.AppendAllTextAsync(summary, coverage.AppendLine().ToString()).ConfigureAwait(false);
        }

        return 0;
    }
}

internal sealed record FoundryMatrix(IReadOnlyList<FoundryPlanEntry> Include);

internal sealed record FoundryPlanEntry(string Runner, string RunnerName, string Family, string Alias,
    string Variant, int FileSizeMb, int RunnerMemoryMb);
