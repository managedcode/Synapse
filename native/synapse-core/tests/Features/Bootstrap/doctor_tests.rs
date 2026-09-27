use synapse_core::bootstrap::{DoctorErrorCode, DoctorRequest, inspect};

#[test]
fn doctor_detects_cpu() {
    let result = inspect(DoctorRequest {
        memory_budget_bytes: 1 << 30,
    });

    assert!(result.supported);
    assert!(!result.architecture.is_empty());
    assert_eq!(result.memory_budget_bytes, 1 << 30);
}

#[test]
fn invalid_budget_fails() {
    let result = inspect(DoctorRequest {
        memory_budget_bytes: 0,
    });

    assert!(!result.supported);
    assert_eq!(
        result.error.expect("typed error").code,
        DoctorErrorCode::InvalidMemoryBudget
    );
}
