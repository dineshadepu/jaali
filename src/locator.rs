use crate::bvh::{Bvh2D, Bvh3D};
use crate::mesh::{TetMesh, TriMesh};
use smallvec::SmallVec;

#[cfg(feature = "gpu")]
use crate::bvh::{Bvh2DGPU, Bvh3DGPU};
#[cfg(feature = "gpu")]
use crate::mesh::{TetMeshGPU, TriMeshGPU};

use crate::gpu::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::sync::Arc;

const MAX_HITS_2D: usize = 32;
const MAX_HITS_3D: usize = 64;

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

    // ---- authoritative locate_all buffers ----
    pub max_hits: usize,
    pub indices: Vec<i32>, // len = queries * max_hits
    pub counts: Vec<u16>,  // len = queries

    #[cfg(feature = "gpu")]
    gpu: Option<Locator2DGPU>,
}

#[cfg(feature = "gpu")]
struct Locator2DGPU {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,

    bvh: Bvh2DGPU,
    mesh: TriMeshGPU,

    indices: CudaSlice<i32>,
    counts: CudaSlice<u16>,
}

impl<'a> Locator2D<'a> {
    /// Minimal constructor (single-hit legacy usage)
    pub fn new(mesh: &'a TriMesh<'a>) -> Self {
        Self::new_with_capacity(mesh, 0, MAX_HITS_2D)
    }

    /// Authoritative constructor
    pub fn new_with_capacity(mesh: &'a TriMesh<'a>, max_queries: usize, max_hits: usize) -> Self {
        assert!(max_hits > 0);

        Self {
            bvh: Bvh2D::build(mesh),
            mesh,
            backend: Backend::Serial,

            max_hits,
            indices: vec![-1; max_queries * max_hits],
            counts: vec![0; max_queries],

            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

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
        // clone for later use
        let stream_for_buffers = stream.clone();

        let mesh_gpu = self.mesh.to_gpu(stream.clone())?;
        let bvh_gpu = self.bvh.to_gpu(stream.clone())?;

        let module = cuda.load_module("cuda_kernels/locate_triangles.ptx")?;
        let kernel = module.get("locate_triangles_all")?;

        self.backend = Backend::GPU;
        self.gpu = Some(Locator2DGPU {
            stream,
            kernel,
            mesh: mesh_gpu,
            bvh: bvh_gpu,
            indices: stream_for_buffers.clone().alloc_zeros::<i32>(0)?,
            counts: stream_for_buffers.clone().alloc_zeros::<u16>(0)?,
        });

        Ok(self)
    }

    #[cfg(not(feature = "gpu"))]
    fn init_gpu_backend(self) -> GpuResult<Self> {
        Err(GpuError::Unavailable)
    }
}

impl<'a> Locator2D<'a> {
    /// Fill internal indices + counts
    pub fn locate_all(&mut self, qx: &[f64], qy: &[f64]) -> GpuResult<()> {
        let n = qx.len();
        let required = n * self.max_hits;

        // -------- CPU resize if needed --------
        if self.indices.len() < required {
            self.indices.resize(required, -1);
        }
        if self.counts.len() < n {
            self.counts.resize(n, 0);
        }

        // -------- GPU resize if needed --------
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            if gpu.indices.len() < required {
                gpu.indices = gpu.stream.alloc_zeros::<i32>(required)?;
            }
            if gpu.counts.len() < n {
                gpu.counts = gpu.stream.alloc_zeros::<u16>(n)?;
            }
        }

        // -------- dispatch backend --------
        match self.backend {
            Backend::Serial => {
                self.locate_all_cpu(qx, qy);
                Ok(())
            }
            Backend::ParallelCPU => {
                self.locate_all_parallel(qx, qy);
                Ok(())
            }
            Backend::GPU => self.locate_all_gpu(qx, qy),
        }
    }
}

impl<'a> Locator2D<'a> {
    fn locate_all_cpu(&mut self, qx: &[f64], qy: &[f64]) {
        let H = self.max_hits;

        for q in 0..qx.len() {
            let base = q * H;

            let (hits, ids) = self.bvh.find_all(qx[q], qy[q], self.mesh, H);

            self.counts[q] = hits as u16;

            for i in 0..hits {
                self.indices[base + i] = ids[i];
            }
        }
    }

    fn locate_all_parallel(&mut self, qx: &[f64], qy: &[f64]) {
        assert_eq!(qx.len(), qy.len());
        let H = self.max_hits;

        #[cfg(feature = "rayon")]
        {
            let mesh = self.mesh;
            let bvh = &self.bvh;

            let results: Vec<(usize, Vec<i32>)> = (0..qx.len())
                .into_par_iter()
                .map(|q| bvh.find_all(qx[q], qy[q], mesh, H))
                .collect();

            for (q, (hits, ids)) in results.into_iter().enumerate() {
                let base = q * H;
                self.counts[q] = hits as u16;

                for i in 0..hits {
                    self.indices[base + i] = ids[i];
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.locate_all_cpu(qx, qy);
        }
    }
    #[cfg(feature = "gpu")]
    fn locate_all_gpu(&mut self, qx: &[f64], qy: &[f64]) -> GpuResult<()> {
        let gpu = self.gpu.as_mut().expect("GPU backend not initialized");

        let n = qx.len();
        let H = self.max_hits;
        let required = n * H;

        debug_assert!(gpu.indices.len() >= required);
        debug_assert!(gpu.counts.len() >= n);

        // -------------------------------------------------
        // Upload queries
        // -------------------------------------------------
        let qx_d = gpu.stream.clone_htod(qx)?;
        let qy_d = gpu.stream.clone_htod(qy)?;

        // -------------------------------------------------
        // Clear counts on GPU (CRITICAL)
        // -------------------------------------------------
        gpu.stream.memset_zeros(&mut gpu.counts)?;

        // -------------------------------------------------
        // Launch kernel
        // -------------------------------------------------
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = gpu.stream.launch_builder(&gpu.kernel);

        launch.arg(&qx_d);
        launch.arg(&qy_d);
        launch.arg(&gpu.indices);
        launch.arg(&gpu.counts);

        let n_i32 = n as i32;
        let h_i32 = H as i32;
        launch.arg(&n_i32);
        launch.arg(&h_i32);

        // BVH
        launch.arg(&gpu.bvh.xmin);
        launch.arg(&gpu.bvh.ymin);
        launch.arg(&gpu.bvh.xmax);
        launch.arg(&gpu.bvh.ymax);
        launch.arg(&gpu.bvh.left);
        launch.arg(&gpu.bvh.right);
        launch.arg(&gpu.bvh.tri);

        // Mesh
        launch.arg(&gpu.mesh.vx);
        launch.arg(&gpu.mesh.vy);
        launch.arg(&gpu.mesh.t0);
        launch.arg(&gpu.mesh.t1);
        launch.arg(&gpu.mesh.t2);

        unsafe { launch.launch(cfg)? };

        // -------------------------------------------------
        // Copy results back (only if CPU needs them)
        // -------------------------------------------------
        gpu.stream.memcpy_dtoh(&gpu.indices, &mut self.indices)?;
        gpu.stream.memcpy_dtoh(&gpu.counts, &mut self.counts)?;

        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn locate_all_gpu(&self, _qx: &[f64], _qy: &[f64]) -> GpuResult<()> {
        Err(GpuError::Unavailable)
    }

    pub fn locate(&mut self, qx: &[f64], qy: &[f64], out: &mut [i32]) {
        assert_eq!(qx.len(), out.len());

        self.locate_all(qx, qy).expect("JAALI locate failed");

        for i in 0..qx.len() {
            out[i] = if self.counts[i] > 0 {
                self.indices[i * self.max_hits]
            } else {
                -1
            };
        }
    }
}

/* ========================== 3D ========================== */
pub struct Locator3D<'a> {
    mesh: &'a TetMesh<'a>,
    bvh: Bvh3D,
    backend: Backend,

    // locate_all storage
    pub max_hits: usize,
    pub indices: Vec<i32>, // size = max_queries * max_hits
    pub counts: Vec<u16>,  // size = max_queries

    #[cfg(feature = "gpu")]
    gpu: Option<Locator3DGPU>,
}

#[cfg(feature = "gpu")]
struct Locator3DGPU {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,

    bvh: Bvh3DGPU,
    mesh: TetMeshGPU,

    indices: CudaSlice<i32>,
    counts: CudaSlice<u16>,
}

impl<'a> Locator3D<'a> {
    /// Minimal constructor (single-hit legacy usage)
    pub fn new(mesh: &'a TetMesh<'a>) -> Self {
        Self::new_with_capacity(mesh, 0, MAX_HITS_3D)
    }
    pub fn new_with_capacity(mesh: &'a TetMesh<'a>, max_queries: usize, max_hits: usize) -> Self {
        Self {
            bvh: Bvh3D::build(mesh),
            mesh,
            backend: Backend::Serial,

            max_hits,
            indices: vec![-1; max_queries * max_hits],
            counts: vec![0; max_queries],

            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

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
        let stream_for_buffers = stream.clone();

        let mesh_gpu = self.mesh.to_gpu(stream.clone())?;
        let bvh_gpu = self.bvh.to_gpu(stream.clone())?;

        let module = cuda.load_module("cuda_kernels/locate_tets.ptx")?;
        let kernel = module.get("locate_tets_all")?;

        self.backend = Backend::GPU;
        self.gpu = Some(Locator3DGPU {
            stream,
            kernel,
            mesh: mesh_gpu,
            bvh: bvh_gpu,
            indices: stream_for_buffers.alloc_zeros::<i32>(0)?,
            counts: stream_for_buffers.alloc_zeros::<u16>(0)?,
        });

        Ok(self)
    }

    #[cfg(not(feature = "gpu"))]
    fn init_gpu_backend(self) -> GpuResult<Self> {
        Err(GpuError::Unavailable)
    }

    pub fn locate_all(&mut self, qx: &[f64], qy: &[f64], qz: &[f64]) -> GpuResult<()> {
        let n = qx.len();
        let required = n * self.max_hits;

        // -------- CPU resize --------
        if self.indices.len() < required {
            self.indices.resize(required, -1);
        }
        if self.counts.len() < n {
            self.counts.resize(n, 0);
        }

        // -------- GPU resize --------
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            if gpu.indices.len() < required {
                gpu.indices = gpu.stream.alloc_zeros::<i32>(required)?;
            }
            if gpu.counts.len() < n {
                gpu.counts = gpu.stream.alloc_zeros::<u16>(n)?;
            }
        }

        match self.backend {
            Backend::Serial => {
                self.locate_all_cpu(qx, qy, qz);
                Ok(())
            }
            Backend::ParallelCPU => {
                self.locate_all_parallel(qx, qy, qz);
                Ok(())
            }
            Backend::GPU => self.locate_all_gpu(qx, qy, qz),
        }
    }

    pub fn locate_all_cpu(&mut self, qx: &[f64], qy: &[f64], qz: &[f64]) {
        assert_eq!(qx.len(), qy.len());
        assert_eq!(qx.len(), qz.len());
        assert!(qx.len() <= self.counts.len());

        let H = self.max_hits;

        for q in 0..qx.len() {
            let base = q * H;

            let (hits, ids) = self.bvh.find_all(qx[q], qy[q], qz[q], self.mesh, H);

            self.counts[q] = hits as u16;

            for i in 0..hits {
                self.indices[base + i] = ids[i];
            }
        }
    }

    pub fn locate_all_parallel(&mut self, qx: &[f64], qy: &[f64], qz: &[f64]) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let H = self.max_hits;
            let mesh = self.mesh;
            let bvh = &self.bvh;

            let results: Vec<(usize, Vec<i32>)> = (0..qx.len())
                .into_par_iter()
                .map(|q| bvh.find_all(qx[q], qy[q], qz[q], mesh, H))
                .collect();

            for (q, (hits, ids)) in results.into_iter().enumerate() {
                let base = q * H;
                self.counts[q] = hits as u16;

                for i in 0..hits {
                    self.indices[base + i] = ids[i];
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.locate_all_cpu(qx, qy, qz);
        }
    }

    #[cfg(feature = "gpu")]
    fn locate_all_gpu(&mut self, qx: &[f64], qy: &[f64], qz: &[f64]) -> GpuResult<()> {
        let gpu = self.gpu.as_mut().expect("GPU backend not initialized");

        let n = qx.len();
        let H = self.max_hits;

        debug_assert!(gpu.indices.len() >= n * H);
        debug_assert!(gpu.counts.len() >= n);

        // Upload queries
        let qx_d = gpu.stream.clone_htod(qx)?;
        let qy_d = gpu.stream.clone_htod(qy)?;
        let qz_d = gpu.stream.clone_htod(qz)?;

        // Clear counts (CRITICAL)
        gpu.stream.memset_zeros(&mut gpu.counts)?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut launch = gpu.stream.launch_builder(&gpu.kernel);

        launch.arg(&qx_d);
        launch.arg(&qy_d);
        launch.arg(&qz_d);
        launch.arg(&gpu.indices);
        launch.arg(&gpu.counts);

        let n_i32 = n as i32;
        let h_i32 = H as i32;
        launch.arg(&n_i32);
        launch.arg(&h_i32);

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

        // Copy back
        gpu.stream.memcpy_dtoh(&gpu.indices, &mut self.indices)?;
        gpu.stream.memcpy_dtoh(&gpu.counts, &mut self.counts)?;

        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn locate_all_gpu(&self, _qx: &[f64], _qy: &[f64], _qz: &[f64]) -> GpuResult<()> {
        Err(GpuError::Unavailable)
    }

    pub fn locate(&mut self, qx: &[f64], qy: &[f64], qz: &[f64], out: &mut [i32]) {
        assert_eq!(qx.len(), out.len());

        self.locate_all(qx, qy, qz).expect("JAALI locate failed");

        for i in 0..qx.len() {
            out[i] = if self.counts[i] > 0 {
                self.indices[i * self.max_hits]
            } else {
                -1
            };
        }
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
