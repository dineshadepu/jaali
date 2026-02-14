use crate::bvh::Bvh2D;
use crate::bvh::Bvh3D;
use crate::mesh::TetMesh;
use crate::mesh::TriMesh;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Serial,
    ParallelCpu,
}

#[derive(Clone, Copy, Debug)]
pub enum LocateMode {
    StrictInside,
    // Planned:
    // NearestCell,
    // Neighborhood(Neighborhood),
}

pub struct Locator2D<'a> {
    mesh: &'a TriMesh<'a>,
    bvh: Bvh2D,
    backend: Backend,
}

impl<'a> Locator2D<'a> {
    /// Build a locator for a given mesh
    pub fn new(mesh: &'a TriMesh<'a>) -> Self {
        Self {
            bvh: Bvh2D::build(mesh),
            mesh,
            backend: Backend::Serial,
        }
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Locate points using default mode (StrictInside)
    #[inline]
    pub fn locate(&self, qx: &[f64], qy: &[f64], out: &mut [i32]) {
        self.locate_with_mode(qx, qy, out, LocateMode::StrictInside);
    }

    /// Locate points with explicit mode
    pub fn locate_with_mode(&self, qx: &[f64], qy: &[f64], out: &mut [i32], mode: LocateMode) {
        assert_eq!(qx.len(), qy.len());
        assert_eq!(qx.len(), out.len());

        match mode {
            LocateMode::StrictInside => {
                self.locate_strict_inside(qx, qy, out);
            }
        }
    }

    #[inline(always)]
    fn locate_strict_inside(&self, qx: &[f64], qy: &[f64], out: &mut [i32]) {
        assert_eq!(qx.len(), qy.len());
        assert_eq!(qx.len(), out.len());

        match self.backend {
            Backend::Serial => self.locate_strict_inside_serial(qx, qy, out),

            Backend::ParallelCpu => self.locate_strict_inside_parallel(qx, qy, out),
        }
    }

    #[inline(always)]
    fn locate_strict_inside_serial(&self, qx: &[f64], qy: &[f64], out: &mut [i32]) {
        for i in 0..qx.len() {
            out[i] = self.bvh.find(qx[i], qy[i], self.mesh);
        }
    }

    #[inline(always)]
    fn locate_strict_inside_parallel(&self, qx: &[f64], qy: &[f64], out: &mut [i32]) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o = self.bvh.find(qx[i], qy[i], self.mesh);
            });
        }

        #[cfg(not(feature = "rayon"))]
        {
            // fallback: behave like serial if rayon not enabled
            self.locate_strict_inside_serial(qx, qy, out);
        }
    }
}

pub struct Locator3D<'a> {
    mesh: &'a TetMesh<'a>,
    bvh: Bvh3D,
    backend: Backend,
}

impl<'a> Locator3D<'a> {
    /// Build a 3D locator (default = serial backend)
    pub fn new(mesh: &'a TetMesh<'a>) -> Self {
        Self {
            bvh: Bvh3D::build(mesh),
            mesh,
            backend: Backend::Serial,
        }
    }

    /// Select execution backend (Serial / ParallelCpu)
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Locate points using default mode (StrictInside)
    #[inline]
    pub fn locate(&self, qx: &[f64], qy: &[f64], qz: &[f64], out: &mut [i32]) {
        self.locate_with_mode(qx, qy, qz, out, LocateMode::StrictInside);
    }

    /// Locate points with explicit mode
    pub fn locate_with_mode(
        &self,
        qx: &[f64],
        qy: &[f64],
        qz: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) {
        assert_eq!(qx.len(), qy.len());
        assert_eq!(qx.len(), qz.len());
        assert_eq!(qx.len(), out.len());

        match mode {
            LocateMode::StrictInside => {
                self.locate_strict_inside(qx, qy, qz, out);
            }
        }
    }

    // --------------------------------------------------------
    // StrictInside
    // --------------------------------------------------------

    #[inline(always)]
    fn locate_strict_inside(&self, qx: &[f64], qy: &[f64], qz: &[f64], out: &mut [i32]) {
        match self.backend {
            Backend::Serial => self.locate_strict_inside_serial(qx, qy, qz, out),

            Backend::ParallelCpu => self.locate_strict_inside_parallel(qx, qy, qz, out),
        }
    }

    #[inline(always)]
    fn locate_strict_inside_serial(&self, qx: &[f64], qy: &[f64], qz: &[f64], out: &mut [i32]) {
        for i in 0..qx.len() {
            out[i] = self.bvh.find(qx[i], qy[i], qz[i], self.mesh);
        }
    }

    #[inline(always)]
    fn locate_strict_inside_parallel(&self, qx: &[f64], qy: &[f64], qz: &[f64], out: &mut [i32]) {
        #[cfg(feature = "rayon")]
        {
            // zip-style parallel traversal (clean + cache-friendly)
            (out, qx, qy, qz)
                .into_par_iter()
                .for_each(|(o, &x, &y, &z)| {
                    *o = self.bvh.find(x, y, z, self.mesh);
                });
        }

        #[cfg(not(feature = "rayon"))]
        {
            // Safety fallback (should not be hit if backend is chosen correctly)
            self.locate_strict_inside_serial(qx, qy, qz, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriMesh;

    #[test]
    fn locator2d_basic() {
        let vx = vec![0.0, 1.0, 0.0];
        let vy = vec![0.0, 0.0, 1.0];
        let t0 = vec![0usize];
        let t1 = vec![1usize];
        let t2 = vec![2usize];

        let mesh = TriMesh {
            vx: &vx,
            vy: &vy,
            t0: &t0,
            t1: &t1,
            t2: &t2,
        };

        let qx = vec![0.25, 1.5];
        let qy = vec![0.25, 1.5];
        let mut out = vec![-99; 2];

        let backends = [Backend::Serial, Backend::ParallelCpu];

        for backend in backends {
            let locator = Locator2D::new(&mesh).with_backend(backend);
            locator.locate(&qx, &qy, &mut out);

            assert_eq!(out, vec![0, -1]);
        }
    }
}

#[cfg(test)]
mod tests_3d {
    use super::*;
    use crate::mesh::TetMesh;

    #[test]
    fn locator3d_basic() {
        // Single tetrahedron
        let vx = vec![0.0, 1.0, 0.0, 0.0];
        let vy = vec![0.0, 0.0, 1.0, 0.0];
        let vz = vec![0.0, 0.0, 0.0, 1.0];

        let t0 = vec![0usize];
        let t1 = vec![1usize];
        let t2 = vec![2usize];
        let t3 = vec![3usize];

        let mesh = TetMesh {
            vx: &vx,
            vy: &vy,
            vz: &vz,
            t0: &t0,
            t1: &t1,
            t2: &t2,
            t3: &t3,
        };

        let qx = vec![0.1, 2.0];
        let qy = vec![0.1, 2.0];
        let qz = vec![0.1, 2.0];

        let mut out = vec![-99; 2];

        let backends = [Backend::Serial, Backend::ParallelCpu];

        for backend in backends {
            let locator = Locator3D::new(&mesh).with_backend(backend);
            locator.locate(&qx, &qy, &qz, &mut out);

            assert_eq!(out, vec![0, -1]);
        }
    }
}
