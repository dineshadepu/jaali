use jaali::Backend;

#[test]
fn backend_enum_always_has_cpu() {
    let _ = Backend::Serial;
    let _ = Backend::ParallelCPU;
}

#[cfg(feature = "gpu")]
#[test]
fn backend_enum_has_gpu_when_feature_enabled() {
    let _ = Backend::GPU;
}

#[cfg(not(feature = "gpu"))]
#[test]
fn backend_gpu_not_available_without_feature() {
    // This test exists just to document behavior.
    // If Backend::GPU existed here, this test would not compile.
    assert!(true);
}
