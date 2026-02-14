use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::locator::{Backend, Locator2D};
use crate::mesh::TriMesh;

#[pyclass]
pub struct PyLocator2D {
    locator: Locator2D<'static>,
}

#[pymethods]
impl PyLocator2D {
    #[new]
    fn new(
        vx: Vec<f64>,
        vy: Vec<f64>,
        t0: Vec<usize>,
        t1: Vec<usize>,
        t2: Vec<usize>,
    ) -> PyResult<Self> {
        // Leak memory intentionally – Python owns lifetime
        let vx = Box::leak(vx.into_boxed_slice());
        let vy = Box::leak(vy.into_boxed_slice());
        let t0 = Box::leak(t0.into_boxed_slice());
        let t1 = Box::leak(t1.into_boxed_slice());
        let t2 = Box::leak(t2.into_boxed_slice());

        let mesh = Box::leak(Box::new(TriMesh { vx, vy, t0, t1, t2 }));

        let locator = Locator2D::new(mesh).with_backend(Backend::ParallelCpu);

        Ok(Self { locator })
    }

    fn locate<'py>(
        &self,
        py: Python<'py>,
        qx: PyReadonlyArray1<f64>,
        qy: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let qx = qx.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("qx must be a contiguous float64 array")
        })?;

        let qy = qy.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("qy must be a contiguous float64 array")
        })?;

        let mut out = vec![-1_i32; qx.len()];
        self.locator.locate(qx, qy, &mut out);

        Ok(PyArray1::from_vec(py, out))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLocator2D>()?;
    Ok(())
}
