use jaali::{TetMesh, TriMesh};

pub fn single_triangle() -> TriMesh<'static> {
    let vx = Box::leak(vec![0.0, 1.0, 0.0].into_boxed_slice());
    let vy = Box::leak(vec![0.0, 0.0, 1.0].into_boxed_slice());

    let t0 = Box::leak(vec![0usize].into_boxed_slice());
    let t1 = Box::leak(vec![1usize].into_boxed_slice());
    let t2 = Box::leak(vec![2usize].into_boxed_slice());

    TriMesh { vx, vy, t0, t1, t2 }
}

pub fn single_tet() -> TetMesh<'static> {
    let vx = Box::leak(vec![0.0, 1.0, 0.0, 0.0].into_boxed_slice());
    let vy = Box::leak(vec![0.0, 0.0, 1.0, 0.0].into_boxed_slice());
    let vz = Box::leak(vec![0.0, 0.0, 0.0, 1.0].into_boxed_slice());

    let t0 = Box::leak(vec![0usize].into_boxed_slice());
    let t1 = Box::leak(vec![1usize].into_boxed_slice());
    let t2 = Box::leak(vec![2usize].into_boxed_slice());
    let t3 = Box::leak(vec![3usize].into_boxed_slice());

    TetMesh {
        vx,
        vy,
        vz,
        t0,
        t1,
        t2,
        t3,
    }
}

pub fn backends() -> Vec<jaali::Backend> {
    let mut b = vec![jaali::Backend::Serial, jaali::Backend::ParallelCPU];
    #[cfg(feature = "gpu")]
    b.push(jaali::Backend::GPU);
    b
}

// ------------------------------------------------------------
// Helpers for stress test
// ------------------------------------------------------------
pub fn brute_force_find_2d(px: f64, py: f64, mesh: &TriMesh) -> i32 {
    for i in 0..mesh.t0.len() {
        let a = mesh.t0[i];
        let b = mesh.t1[i];
        let c = mesh.t2[i];

        if jaali::geometry::point_in_triangle_strict(
            px, py, mesh.vx[a], mesh.vy[a], mesh.vx[b], mesh.vy[b], mesh.vx[c], mesh.vy[c],
        ) {
            return i as i32;
        }
    }
    -1
}

pub fn generate_grid_mesh_2d(
    nx: usize,
    ny: usize,
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut vx = Vec::with_capacity(nx * ny);
    let mut vy = Vec::with_capacity(nx * ny);

    for j in 0..ny {
        for i in 0..nx {
            vx.push(i as f64);
            vy.push(j as f64);
        }
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();

    let idx = |i: usize, j: usize| j * nx + i;

    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let v00 = idx(i, j);
            let v10 = idx(i + 1, j);
            let v01 = idx(i, j + 1);
            let v11 = idx(i + 1, j + 1);

            t0.push(v00);
            t1.push(v10);
            t2.push(v11);

            t0.push(v00);
            t1.push(v11);
            t2.push(v01);
        }
    }

    (vx, vy, t0, t1, t2)
}

// ------------------------------------------------------------
// Helpers for stress test
// ------------------------------------------------------------
pub fn brute_force_find_3d(px: f64, py: f64, pz: f64, mesh: &TetMesh) -> i32 {
    for i in 0..mesh.t0.len() {
        let a = mesh.t0[i];
        let b = mesh.t1[i];
        let c = mesh.t2[i];
        let d = mesh.t3[i];

        if jaali::geometry::point_in_tet_strict(
            px, py, pz, mesh.vx[a], mesh.vy[a], mesh.vz[a], mesh.vx[b], mesh.vy[b], mesh.vz[b],
            mesh.vx[c], mesh.vy[c], mesh.vz[c], mesh.vx[d], mesh.vy[d], mesh.vz[d],
        ) {
            return i as i32;
        }
    }
    -1
}
