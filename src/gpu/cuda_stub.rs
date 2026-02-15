// src/gpu/cuda_stub.rs
#[derive(Debug, Clone)]
pub enum GpuError {
    /// GPU support not compiled in
    Unavailable,
}

pub type GpuResult<T> = Result<T, GpuError>;

#[derive(Clone)]
pub struct CudaManager;

#[derive(Clone)]
pub struct CudaStream;

#[derive(Clone)]
pub struct CudaFunction;

#[derive(Clone)]
pub struct ModuleHandle;

impl CudaManager {
    pub fn new(_: usize) -> GpuResult<Self> {
        Err(GpuError::Unavailable)
    }

    pub fn new_stream(&self) -> GpuResult<std::sync::Arc<CudaStream>> {
        Err(GpuError::Unavailable)
    }

    pub fn load_module(&self, _: &str) -> GpuResult<ModuleHandle> {
        Err(GpuError::Unavailable)
    }
}

impl ModuleHandle {
    pub fn get(&self, _: &str) -> GpuResult<CudaFunction> {
        Err(GpuError::Unavailable)
    }
}
