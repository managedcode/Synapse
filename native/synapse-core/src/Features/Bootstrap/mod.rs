use std::ffi::OsString;
use std::process::ExitCode;

const SUCCESS_EXIT_CODE: u8 = 0;
const INVALID_CONFIGURATION_EXIT_CODE: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DoctorRequest {
    pub memory_budget_bytes: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoctorErrorCode {
    InvalidMemoryBudget,
    UnsupportedArchitecture,
}

impl DoctorErrorCode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidMemoryBudget => "invalidMemoryBudget",
            Self::UnsupportedArchitecture => "unsupportedArchitecture",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoctorError {
    pub code: DoctorErrorCode,
    pub message: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuCapabilities {
    pub neon: bool,
    pub avx2: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoctorResult {
    pub supported: bool,
    pub architecture: &'static str,
    pub operating_system: &'static str,
    pub memory_budget_bytes: i64,
    pub cpu: CpuCapabilities,
    pub error: Option<DoctorError>,
}

#[must_use]
pub fn inspect(request: DoctorRequest) -> DoctorResult {
    let architecture = std::env::consts::ARCH;
    let operating_system = std::env::consts::OS;
    let cpu = CpuCapabilities {
        neon: has_neon(),
        avx2: has_avx2(),
    };

    if request.memory_budget_bytes <= 0 {
        return failure(
            request,
            architecture,
            operating_system,
            cpu,
            DoctorErrorCode::InvalidMemoryBudget,
            "memory budget must be a positive signed 64-bit byte count",
        );
    }

    if architecture != "aarch64" && architecture != "x86_64" {
        return failure(
            request,
            architecture,
            operating_system,
            cpu,
            DoctorErrorCode::UnsupportedArchitecture,
            "architecture is not supported by the bootstrap CPU profile",
        );
    }

    DoctorResult {
        supported: true,
        architecture,
        operating_system,
        memory_budget_bytes: request.memory_budget_bytes,
        cpu,
        error: None,
    }
}

#[must_use]
pub fn run_cli<I>(arguments: I) -> ExitCode
where
    I: IntoIterator<Item = OsString>,
{
    match parse_request(arguments) {
        Ok(request) => {
            let result = inspect(request);
            println!("{}", result.to_json());
            if result.supported {
                ExitCode::from(SUCCESS_EXIT_CODE)
            } else {
                ExitCode::from(INVALID_CONFIGURATION_EXIT_CODE)
            }
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(INVALID_CONFIGURATION_EXIT_CODE)
        }
    }
}

fn parse_request<I>(arguments: I) -> Result<DoctorRequest, &'static str>
where
    I: IntoIterator<Item = OsString>,
{
    let mut arguments = arguments.into_iter();
    let _executable = arguments.next();
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("doctor")) {
        return Err("usage: synapse-runtime doctor --memory-budget-bytes <bytes>");
    }

    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("--memory-budget-bytes")) {
        return Err("missing --memory-budget-bytes");
    }

    let raw_budget = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or("memory budget must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("unexpected trailing doctor arguments");
    }

    let memory_budget_bytes = raw_budget
        .parse::<i64>()
        .map_err(|_| "memory budget must be a signed 64-bit integer")?;
    Ok(DoctorRequest {
        memory_budget_bytes,
    })
}

impl DoctorResult {
    fn to_json(&self) -> String {
        let error = self.error.as_ref().map_or_else(
            || "null".to_owned(),
            |error| {
                format!(
                    "{{\"code\":\"{}\",\"message\":\"{}\"}}",
                    error.code.as_str(),
                    error.message
                )
            },
        );

        format!(
            concat!(
                "{{\"supported\":{},",
                "\"architecture\":\"{}\",",
                "\"operatingSystem\":\"{}\",",
                "\"memoryBudgetBytes\":{},",
                "\"cpu\":{{\"neon\":{},\"avx2\":{}}},",
                "\"error\":{}}}"
            ),
            self.supported,
            self.architecture,
            self.operating_system,
            self.memory_budget_bytes,
            self.cpu.neon,
            self.cpu.avx2,
            error
        )
    }
}

const fn failure(
    request: DoctorRequest,
    architecture: &'static str,
    operating_system: &'static str,
    cpu: CpuCapabilities,
    code: DoctorErrorCode,
    message: &'static str,
) -> DoctorResult {
    DoctorResult {
        supported: false,
        architecture,
        operating_system,
        memory_budget_bytes: request.memory_budget_bytes,
        cpu,
        error: Some(DoctorError { code, message }),
    }
}

#[cfg(target_arch = "aarch64")]
fn has_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
const fn has_neon() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
const fn has_avx2() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::{DoctorRequest, inspect};

    #[test]
    fn positive_budget_reports_current_platform() {
        let result = inspect(DoctorRequest {
            memory_budget_bytes: 4096,
        });

        assert_eq!(result.architecture, std::env::consts::ARCH);
        assert_eq!(result.operating_system, std::env::consts::OS);
    }
}
