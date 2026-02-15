use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::locator::{Backend, Locator2D, Locator3D};
use crate::mesh::{TetMesh, TriMesh};

/// ---------------------------------------------
/// Backend parsing (shared by 2D and 3D)
/// ---------------------------------------------
fn parse_backend(backend: Option<&str>) -> PyResult<Backend> {
    match backend.unwrap_or("parallel") {
        "serial" => Ok(Backend::Serial),
        "parallel" | "rayon" => Ok(Backend::ParallelCPU),

        #[cfg(feature = "gpu")]
        "gpu" => Ok(Backend::GPU),

        #[cfg(not(feature = "gpu"))]
        "gpu" => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "JAALI was built without GPU support",
        )),

        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown backend '{}'. Expected 'serial', 'parallel', or 'gpu'",
            other
        ))),
    }
}

#[pyclass]
pub struct PyLocator2D {
    locator: Locator2D<'static>,
}

#[pymethods]
impl PyLocator2D {
    #[new]
    #[pyo3(signature = (vx, vy, t0, t1, t2, backend=None))]
    fn new(
        vx: Vec<f64>,
        vy: Vec<f64>,
        t0: Vec<usize>,
        t1: Vec<usize>,
        t2: Vec<usize>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        // Leak memory intentionally – Python owns lifetime
        let vx = Box::leak(vx.into_boxed_slice());
        let vy = Box::leak(vy.into_boxed_slice());
        let t0 = Box::leak(t0.into_boxed_slice());
        let t1 = Box::leak(t1.into_boxed_slice());
        let t2 = Box::leak(t2.into_boxed_slice());

        let mesh = Box::leak(Box::new(TriMesh { vx, vy, t0, t1, t2 }));

        let backend = parse_backend(backend)?;
        let locator = Locator2D::new(mesh).with_backend(backend).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize backend {:?}: {:?}",
                backend, e
            ))
        })?;

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

        if qx.len() != qy.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "qx and qy must have the same length",
            ));
        }

        let mut out = vec![-1_i32; qx.len()];
        self.locator.locate(qx, qy, &mut out);

        Ok(PyArray1::from_vec(py, out))
    }
}

#[pyclass]
pub struct PyLocator3D {
    locator: Locator3D<'static>,
}

#[pymethods]
impl PyLocator3D {
    #[new]
    #[pyo3(signature = (vx, vy, vz, t0, t1, t2, t3, backend=None))]
    fn new(
        vx: Vec<f64>,
        vy: Vec<f64>,
        vz: Vec<f64>,
        t0: Vec<usize>,
        t1: Vec<usize>,
        t2: Vec<usize>,
        t3: Vec<usize>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        // -------------------------------------------------
        // Leak memory intentionally — Python owns lifetime
        // -------------------------------------------------
        let vx = Box::leak(vx.into_boxed_slice());
        let vy = Box::leak(vy.into_boxed_slice());
        let vz = Box::leak(vz.into_boxed_slice());

        let t0 = Box::leak(t0.into_boxed_slice());
        let t1 = Box::leak(t1.into_boxed_slice());
        let t2 = Box::leak(t2.into_boxed_slice());
        let t3 = Box::leak(t3.into_boxed_slice());

        let mesh = Box::leak(Box::new(TetMesh {
            vx,
            vy,
            vz,
            t0,
            t1,
            t2,
            t3,
        }));

        let backend = parse_backend(backend)?;
        let locator = Locator3D::new(mesh).with_backend(backend).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize backend {:?}: {:?}",
                backend, e
            ))
        })?;

        Ok(Self { locator })
    }

    fn locate<'py>(
        &self,
        py: Python<'py>,
        qx: PyReadonlyArray1<f64>,
        qy: PyReadonlyArray1<f64>,
        qz: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let qx = qx.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("qx must be a contiguous float64 array")
        })?;

        let qy = qy.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("qy must be a contiguous float64 array")
        })?;

        let qz = qz.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("qz must be a contiguous float64 array")
        })?;

        if qx.len() != qy.len() || qx.len() != qz.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "qx, qy, qz must have the same length",
            ));
        }

        let mut out = vec![-1_i32; qx.len()];
        self.locator.locate(qx, qy, qz, &mut out);

        Ok(PyArray1::from_vec(py, out))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLocator2D>()?;
    m.add_class::<PyLocator3D>()?;
    Ok(())
}
