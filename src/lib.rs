//! JAALI: Joint Algorithm for Adaptive Localization across Interfaces
//!
//! High-performance particle–mesh localization on CPU (GPU support optional).
//!
//! Core idea:
//!   Given a mesh (structured or unstructured) and query points,
//!   efficiently determine which cell contains each point.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bvh;
pub mod geometry;
pub mod locator;
pub mod mesh;
#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod test_bvh_2d;
mod test_bvh_3d;

pub use crate::bvh::{Bvh2D, Bvh3D};
pub use crate::locator::{Backend, Locator2D, Locator3D};
pub use crate::mesh::{TetMesh, TriMesh};
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn jaali(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
