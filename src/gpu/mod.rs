// src/gpu/mod.rs
#![allow(dead_code)]

#[cfg(feature = "gpu")]
pub mod cuda;

#[cfg(feature = "gpu")]
pub use cuda::*;

#[cfg(not(feature = "gpu"))]
pub mod cuda_stub;

#[cfg(not(feature = "gpu"))]
pub use cuda_stub::*;
