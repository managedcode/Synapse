using ManagedCode.Synapse.Cli.Features.Bootstrap;
using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.Cli.Features.TextGeneration;

if (args.Length == 0)
{
    Console.Error.WriteLine(
        "Usage: synapse <doctor|generate|model> [options]");
    return 2;
}

return args[0] switch
{
    "doctor" => DoctorCommand.Run(args[1..]),
    "generate" => await GenerationCommand.RunAsync(args[1..]),
    "model" => await ModelCommand.RunAsync(args[1..]),
    _ => 2,
};
