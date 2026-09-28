using ManagedCode.Synapse.Cli.Features.Bootstrap;
using ManagedCode.Synapse.Cli.Features.LongContext;
using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Cli.Features.Tokenization;

if (args.Length == 0)
{
    Console.Error.WriteLine(
        "Usage: synapse <doctor|generate|model|score|tokenize|detokenize> [options]");
    return 2;
}

return args[0] switch
{
    "doctor" => DoctorCommand.Run(args[1..]),
    "generate" => await GenerationCommand.RunAsync(args[1..]),
    "model" => await ModelCommand.RunAsync(args[1..]),
    "score" => ScoreCommand.Run(args[1..]),
    "tokenize" or "detokenize" => TokenizeCommand.Run(args[0], args[1..]),
    _ => 2,
};
