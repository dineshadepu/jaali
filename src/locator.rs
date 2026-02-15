use crate::bvh::{Bvh2D, Bvh3D};
use crate::mesh::{TetMesh, TriMesh};

#[cfg(feature = "gpu")]
use crate::bvh::{Bvh2DGPU, Bvh3DGPU};
#[cfg(feature = "gpu")]
use crate::mesh::{TetMeshGPU, TriMeshGPU};

use crate::gpu::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backend {
    Serial,
    ParallelCPU,
    GPU,
}

#[derive(Clone, Copy, Debug)]
pub enum LocateMode {
    StrictInside,
    InsideOrBoundary,
}

/* ========================== 2D ========================== */

pub struct Locator2D<'a> {
    mesh: &'a TriMesh<'a>,
    bvh: Bvh2D,
    backend: Backend,

    #[cfg(feature = "gpu")]
    gpu: Option<Locator2DGPU>,
}

#[cfg(feature = "gpu")]
struct Locator2DGPU {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    bvh: Bvh2DGPU,
    mesh: TriMeshGPU,
}

impl<'a> Locator2D<'a> {
    pub fn new(mesh: &'a TriMesh<'a>) -> Self {
        Self {
            bvh: Bvh2D::build(mesh),
            mesh,
            backend: Backend::Serial,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /* -------- backend selection (ALWAYS EXISTS) -------- */

    pub fn with_backend(mut self, backend: Backend) -> GpuResult<Self> {
        match backend {
            Backend::GPU => self.init_gpu_backend(),
            _ => {
                self.backend = backend;
                Ok(self)
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn init_gpu_backend(mut self) -> GpuResult<Self> {
        let cuda = CudaManager::new(0)?;
        let stream = cuda.new_stream()?;

        let mesh_gpu = self.mesh.to_gpu(stream.clone())?;
        let bvh_gpu = self.bvh.to_gpu(stream.clone())?;

        let module = cuda.load_module("cuda_kernels/locate_triangles.ptx")?;
        let kernel = module.get("locate_triangles")?;

        self.backend = Backend::GPU;
        self.gpu = Some(Locator2DGPU {
            stream,
            kernel,
            mesh: mesh_gpu,
            bvh: bvh_gpu,
        });

        Ok(self)
    }

    #[cfg(not(feature = "gpu"))]
    fn init_gpu_backend(self) -> GpuResult<Self> {
        Err(GpuError::Unavailable)
    }

    /* -------- public API -------- */

    pub fn locate(&self, qx: &[f64], qy: &[f64], out: &mut [i32]) {
        if let Err(_) = self.locate_with_mode(qx, qy, out, LocateMode::InsideOrBoundary) {
            panic!("JAALI locate failed");
        }
    }

    pub fn locate_with_mode(
        &self,
        qx: &[f64],
        qy: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) -> GpuResult<()> {
        assert_eq!(qx.len(), qy.len());
        assert_eq!(qx.len(), out.len());
        match mode {
            LocateMode::StrictInside | LocateMode::InsideOrBoundary => {
                self.locate_with_mode_impl(qx, qy, out, mode)
            }
        }
    }
    fn locate_with_mode_impl(
        &self,
        qx: &[f64],
        qy: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) -> GpuResult<()> {
        match self.backend {
            Backend::Serial => {
                for i in 0..qx.len() {
                    out[i] = self.bvh.find(qx[i], qy[i], self.mesh, mode);
                }
                Ok(())
            }

            Backend::ParallelCPU => {
                #[cfg(feature = "rayon")]
                {
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        *o = self.bvh.find(qx[i], qy[i], self.mesh, mode);
                    });
                }

                #[cfg(not(feature = "rayon"))]
                {
                    for i in 0..qx.len() {
                        out[i] = self.bvh.find(qx[i], qy[i], self.mesh, mode);
                    }
                }
                Ok(())
            }

            Backend::GPU => self.locate_gpu(qx, qy, out, mode),
        }
    }

    #[cfg(feature = "gpu")]
    #[allow(unsafe_code)]
    fn locate_gpu(
        &self,
        qx: &[f64],
        qy: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) -> GpuResult<()> {
        let gpu = self.gpu.as_ref().expect("GPU not initialized");

        let n = qx.len();
        let qx_d = gpu.stream.clone_htod(qx)?;
        let qy_d = gpu.stream.clone_htod(qy)?;
        let mut out_d = gpu.stream.alloc_zeros::<i32>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = gpu.stream.launch_builder(&gpu.kernel);

        launch.arg(&qx_d);
        launch.arg(&qy_d);
        launch.arg(&out_d);
        let n_i32 = n as i32;
        launch.arg(&n_i32);

        let mode_i32 = match mode {
            LocateMode::StrictInside => 0,
            LocateMode::InsideOrBoundary => 1,
        };
        launch.arg(&mode_i32);

        launch.arg(&gpu.bvh.xmin);
        launch.arg(&gpu.bvh.ymin);
        launch.arg(&gpu.bvh.xmax);
        launch.arg(&gpu.bvh.ymax);
        launch.arg(&gpu.bvh.left);
        launch.arg(&gpu.bvh.right);
        launch.arg(&gpu.bvh.tri);

        launch.arg(&gpu.mesh.vx);
        launch.arg(&gpu.mesh.vy);
        launch.arg(&gpu.mesh.t0);
        launch.arg(&gpu.mesh.t1);
        launch.arg(&gpu.mesh.t2);

        unsafe { launch.launch(cfg)? };
        gpu.stream.memcpy_dtoh(&out_d, out)?;
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn locate_gpu(&self, _: &[f64], _: &[f64], _: &mut [i32], _: LocateMode) -> GpuResult<()> {
        Err(GpuError::Unavailable)
    }
}

/* ========================== 3D ========================== */
pub struct Locator3D<'a> {
    mesh: &'a TetMesh<'a>,
    bvh: Bvh3D,
    backend: Backend,
    #[cfg(feature = "gpu")]
    gpu: Option<Locator3DGPU>,
}

#[cfg(feature = "gpu")]
pub struct Locator3DGPU {
    pub stream: Arc<CudaStream>,
    pub kernel: CudaFunction,
    pub bvh: Bvh3DGPU,
    pub mesh: TetMeshGPU,
}

impl<'a> Locator3D<'a> {
    pub fn new(mesh: &'a TetMesh<'a>) -> Self {
        Self {
            bvh: Bvh3D::build(mesh),
            mesh,
            backend: Backend::Serial,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /* -------- backend selection (ALWAYS EXISTS) -------- */
    /// Select execution backend (Serial / ParallelCPU / GPU)

    pub fn with_backend(mut self, backend: Backend) -> GpuResult<Self> {
        match backend {
            Backend::GPU => self.init_gpu_backend(),
            _ => {
                self.backend = backend;
                Ok(self)
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn init_gpu_backend(mut self) -> GpuResult<Self> {
        let cuda = CudaManager::new(0)?;
        let stream = cuda.new_stream()?;

        let mesh_gpu = self.mesh.to_gpu(stream.clone())?;
        let bvh_gpu = self.bvh.to_gpu(stream.clone())?;

        let module = cuda.load_module("cuda_kernels/locate_tets.ptx")?;
        let kernel = module.get("locate_tets")?;

        self.backend = Backend::GPU;
        self.gpu = Some(Locator3DGPU {
            stream,
            kernel,
            mesh: mesh_gpu,
            bvh: bvh_gpu,
        });

        Ok(self)
    }

    #[cfg(not(feature = "gpu"))]
    fn init_gpu_backend(self) -> GpuResult<Self> {
        Err(GpuError::Unavailable)
    }

    /* -------- public API -------- */
    /// Locate points using default mode (StrictInside)
    pub fn locate(&self, qx: &[f64], qy: &[f64], qz: &[f64], out: &mut [i32]) {
        if let Err(_) = self.locate_with_mode(qx, qy, qz, out, LocateMode::InsideOrBoundary) {
            panic!("JAALI locate failed");
        }
    }

    /// Locate points with explicit mode
    pub fn locate_with_mode(
        &self,
        qx: &[f64],
        qy: &[f64],
        qz: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) -> GpuResult<()> {
        assert_eq!(qx.len(), qy.len());
        assert_eq!(qx.len(), qz.len());
        assert_eq!(qx.len(), out.len());

        match mode {
            LocateMode::StrictInside | LocateMode::InsideOrBoundary => {
                self.locate_with_mode_impl(qx, qy, qz, out, mode)
            }
        }
    }

    fn locate_with_mode_impl(
        &self,
        qx: &[f64],
        qy: &[f64],
        qz: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) -> GpuResult<()> {
        match self.backend {
            Backend::Serial => {
                for i in 0..qx.len() {
                    out[i] = self.bvh.find(qx[i], qy[i], qz[i], self.mesh, mode);
                }
                Ok(())
            }

            Backend::ParallelCPU => {
                #[cfg(feature = "rayon")]
                {
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        *o = self.bvh.find(qx[i], qy[i], qz[i], self.mesh, mode);
                    });
                }

                #[cfg(not(feature = "rayon"))]
                {
                    for i in 0..qx.len() {
                        out[i] = self.bvh.find(qx[i], qy[i], qz[i], self.mesh, mode);
                    }
                }
                Ok(())
            }

            Backend::GPU => self.locate_gpu(qx, qy, qz, out, mode),
        }
    }

    #[cfg(feature = "gpu")]
    #[allow(unsafe_code)]
    fn locate_gpu(
        &self,
        qx: &[f64],
        qy: &[f64],
        qz: &[f64],
        out: &mut [i32],
        mode: LocateMode,
    ) -> GpuResult<()> {
        let gpu = self.gpu.as_ref().expect("GPU backend not initialized");

        let n = qx.len();

        let qx_d = gpu.stream.clone_htod(qx)?;
        let qy_d = gpu.stream.clone_htod(qy)?;
        let qz_d = gpu.stream.clone_htod(qz)?;
        let mut out_d = gpu.stream.alloc_zeros::<i32>(n)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = gpu.stream.launch_builder(&gpu.kernel);

        launch.arg(&qx_d);
        launch.arg(&qy_d);
        launch.arg(&qz_d);
        launch.arg(&out_d);
        let n_i32 = n as i32;
        launch.arg(&n_i32);

        let mode_i32 = match mode {
            LocateMode::StrictInside => 0,
            LocateMode::InsideOrBoundary => 1,
        };
        launch.arg(&mode_i32);

        // BVH
        launch.arg(&gpu.bvh.xmin);
        launch.arg(&gpu.bvh.ymin);
        launch.arg(&gpu.bvh.zmin);
        launch.arg(&gpu.bvh.xmax);
        launch.arg(&gpu.bvh.ymax);
        launch.arg(&gpu.bvh.zmax);
        launch.arg(&gpu.bvh.left);
        launch.arg(&gpu.bvh.right);
        launch.arg(&gpu.bvh.tet);

        // Mesh
        launch.arg(&gpu.mesh.vx);
        launch.arg(&gpu.mesh.vy);
        launch.arg(&gpu.mesh.vz);
        launch.arg(&gpu.mesh.t0);
        launch.arg(&gpu.mesh.t1);
        launch.arg(&gpu.mesh.t2);
        launch.arg(&gpu.mesh.t3);

        unsafe { launch.launch(cfg)? };
        gpu.stream.memcpy_dtoh(&out_d, out)?;

        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn locate_gpu(
        &self,
        _qx: &[f64],
        _qy: &[f64],
        _qz: &[f64],
        _out: &mut [i32],
        _mode: LocateMode,
    ) -> GpuResult<()> {
        Err(GpuError::Unavailable)
    }
}

// #[cfg(test)]
// mod stress_locator_2d {
//     use super::*;
//     use crate::test_bvh_2d::brute_force_find as brute_force_find_2d;
//     use crate::test_bvh_2d::generate_points_2d;
//     use crate::test_bvh_2d::read_vtk_2d;
//     use std::time::Instant;

//     #[test]
//     #[ignore]
//     fn stress_locator_vs_bruteforce_vtk_2d() {
//         let vtk_path = "./test_data/field_2d.vtk";
//         if !std::path::Path::new(vtk_path).exists() {
//             eprintln!("VTK file not found, skipping stress test");
//             return;
//         }

//         // ----------------------------
//         // Load mesh
//         // ----------------------------
//         let (vx, vy, t0, t1, t2) = read_vtk_2d(vtk_path);

//         let mesh = TriMesh {
//             vx: &vx,
//             vy: &vy,
//             t0: &t0,
//             t1: &t1,
//             t2: &t2,
//         };

//         let n_queries = 100_000;
//         let queries = generate_points_2d(n_queries, &vx, &vy);

//         // ----------------------------
//         // Brute-force reference
//         // ----------------------------
//         let t0 = Instant::now();
//         let brute: Vec<i32> = queries
//             .iter()
//             .map(|&(x, y)| brute_force_find_2d(x, y, &mesh))
//             .collect();
//         let t_brute = t0.elapsed();

//         // ----------------------------
//         // Test all backends
//         // ----------------------------
//         let backends = vec![
//             Backend::Serial,
//             Backend::ParallelCPU,
//             #[cfg(feature = "gpu")]
//             Backend::GPU,
//         ];

//         for backend in backends {
//             let locator = Locator2D::new(&mesh)
//                 .with_backend(backend)
//                 .expect("backend init failed");

//             let mut out = vec![-99; n_queries];

//             let t0 = Instant::now();
//             let (qx, qy): (Vec<_>, Vec<_>) = queries.iter().cloned().unzip();
//             locator.locate(&qx, &qy, &mut out);
//             let t_loc = t0.elapsed();

//             // ----------------------------
//             // Correctness: inside / outside
//             // ----------------------------
//             let mismatches = out
//                 .iter()
//                 .zip(&brute)
//                 .filter(|(a, b)| (**a >= 0) != (**b >= 0))
//                 .count();

//             assert_eq!(mismatches, 0, "Mismatch for backend {:?}", backend);

//             println!(
//                 "{:?}: {:.3} s ({:.2} M q/s)",
//                 backend,
//                 t_loc.as_secs_f64(),
//                 n_queries as f64 / t_loc.as_secs_f64() / 1e6
//             );
//         }

//         println!(
//             "Brute force: {:.3} s ({:.2} M q/s)",
//             t_brute.as_secs_f64(),
//             n_queries as f64 / t_brute.as_secs_f64() / 1e6
//         );
//     }
// }
