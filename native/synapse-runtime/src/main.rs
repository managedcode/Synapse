fn main() -> std::process::ExitCode {
    synapse_core::bootstrap::run_cli(std::env::args_os())
}
