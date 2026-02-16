use jaali::{Backend, Locator3D, TetMesh, TriMesh};

use std::collections::HashSet;

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

pub fn grid_mesh_3d(
    nx: usize,
    ny: usize,
    nz: usize,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
) {
    let mut vx = Vec::new();
    let mut vy = Vec::new();
    let mut vz = Vec::new();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                vx.push(i as f64);
                vy.push(j as f64);
                vz.push(k as f64);
            }
        }
    }

    let idx = |i, j, k| k * nx * ny + j * nx + i;

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();
    let mut t3 = Vec::new();

    for k in 0..nz - 1 {
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let v000 = idx(i, j, k);
                let v100 = idx(i + 1, j, k);
                let v010 = idx(i, j + 1, k);
                let v001 = idx(i, j, k + 1);

                // simple tetra split
                t0.push(v000);
                t1.push(v100);
                t2.push(v010);
                t3.push(v001);
            }
        }
    }

    (vx, vy, vz, t0, t1, t2, t3)
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

fn run_locator3d_and_collect_ids(
    mesh: &TetMesh,
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
    max_hits: usize,
) -> std::collections::HashSet<i32> {
    let mut locator = Locator3D::new(mesh).with_backend(Backend::Serial).unwrap();

    locator.locate_all(qx, qy, qz).unwrap();

    let h = locator.max_hits;
    let count = locator.counts[0] as usize;

    locator.indices[0 * h..0 * h + count]
        .iter()
        .copied()
        .collect()
}

pub fn collect_hits_per_query(
    indices: &[i32],
    counts: &[u16],
    max_hits: usize,
) -> Vec<std::collections::HashSet<i32>> {
    let mut result = Vec::with_capacity(counts.len());

    for q in 0..counts.len() {
        let c = counts[q] as usize;
        let base = q * max_hits;

        let set = indices[base..base + c]
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();

        result.push(set);
    }

    result
}

pub fn assert_locator3d_backends_agree(mesh: &TetMesh, qx: &[f64], qy: &[f64], qz: &[f64]) {
    assert_eq!(qx.len(), qy.len());
    assert_eq!(qx.len(), qz.len());
    let n = qx.len();

    // ---------- Serial ----------
    let mut loc_serial = Locator3D::new(mesh).with_backend(Backend::Serial).unwrap();

    loc_serial.locate_all(qx, qy, qz).unwrap();

    let serial_hits =
        collect_hits_per_query(&loc_serial.indices, &loc_serial.counts, loc_serial.max_hits);

    // ---------- Parallel CPU ----------
    let mut loc_parallel = Locator3D::new(mesh)
        .with_backend(Backend::ParallelCPU)
        .unwrap();

    loc_parallel.locate_all(qx, qy, qz).unwrap();

    let parallel_hits = collect_hits_per_query(
        &loc_parallel.indices,
        &loc_parallel.counts,
        loc_parallel.max_hits,
    );

    assert_eq!(
        loc_serial.counts, loc_parallel.counts,
        "Serial and Parallel counts differ"
    );

    assert_eq!(
        serial_hits, parallel_hits,
        "Serial and Parallel hit sets differ"
    );

    // ---------- GPU ----------
    #[cfg(feature = "gpu")]
    {
        let mut loc_gpu = Locator3D::new(mesh).with_backend(Backend::GPU).unwrap();

        loc_gpu.locate_all(qx, qy, qz).unwrap();

        let gpu_hits = collect_hits_per_query(&loc_gpu.indices, &loc_gpu.counts, loc_gpu.max_hits);

        assert_eq!(
            loc_serial.counts, loc_gpu.counts,
            "CPU and GPU counts differ"
        );

        assert_eq!(serial_hits, gpu_hits, "CPU and GPU hit sets differ");
    }
}

pub fn assert_single_query_expected_count(mesh: &TetMesh, q: [f64; 3], expected: usize) {
    let qx = vec![q[0]];
    let qy = vec![q[1]];
    let qz = vec![q[2]];

    let mut loc = Locator3D::new_with_capacity(mesh, 1, 32)
        .with_backend(Backend::Serial)
        .unwrap();

    loc.locate_all(&qx, &qy, &qz).unwrap();
    assert_eq!(loc.counts[0] as usize, expected);
}

pub fn make_large_unstructured_tet_mesh() -> TetMesh<'static> {
    let nx = 20;
    let ny = 20;
    let nz = 20;

    let mut vx = Vec::new();
    let mut vy = Vec::new();
    let mut vz = Vec::new();

    let idx = |i: usize, j: usize, k: usize| -> usize { i + (nx + 1) * (j + (ny + 1) * k) };

    // vertices
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                vx.push(i as f64 / nx as f64);
                vy.push(j as f64 / ny as f64);
                vz.push(k as f64 / nz as f64);
            }
        }
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();
    let mut t3 = Vec::new();

    // cube → 5 tets
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let v000 = idx(i, j, k);
                let v100 = idx(i + 1, j, k);
                let v010 = idx(i, j + 1, k);
                let v110 = idx(i + 1, j + 1, k);
                let v001 = idx(i, j, k + 1);
                let v101 = idx(i + 1, j, k + 1);
                let v011 = idx(i, j + 1, k + 1);
                let v111 = idx(i + 1, j + 1, k + 1);

                // standard 5-tet split
                t0.extend_from_slice(&[v000, v100, v010, v001, v110]);
                t1.extend_from_slice(&[v100, v110, v010, v111, v101]);
                t2.extend_from_slice(&[v010, v110, v111, v011, v001]);
                t3.extend_from_slice(&[v100, v101, v001, v111, v110]);
            }
        }
    }

    TetMesh {
        vx: Box::leak(vx.into_boxed_slice()),
        vy: Box::leak(vy.into_boxed_slice()),
        vz: Box::leak(vz.into_boxed_slice()),
        t0: Box::leak(t0.into_boxed_slice()),
        t1: Box::leak(t1.into_boxed_slice()),
        t2: Box::leak(t2.into_boxed_slice()),
        t3: Box::leak(t3.into_boxed_slice()),
    }
}

pub fn collect_hits_2d(
    indices: &[i32],
    counts: &[u16],
    max_hits: usize,
    n: usize,
) -> Vec<HashSet<i32>> {
    let mut out = Vec::with_capacity(n);
    for q in 0..n {
        let c = counts[q] as usize;
        let base = q * max_hits;
        out.push(indices[base..base + c].iter().copied().collect());
    }
    out
}

pub fn make_large_unstructured_tri_mesh_2d(nx: usize, ny: usize) -> TriMesh<'static> {
    let mut vx = Vec::new();
    let mut vy = Vec::new();

    let idx = |i: usize, j: usize| -> usize { i + (nx + 1) * j };

    // vertices
    for j in 0..=ny {
        for i in 0..=nx {
            vx.push(i as f64);
            vy.push(j as f64);
        }
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();

    // split each quad into 2 triangles
    for j in 0..ny {
        for i in 0..nx {
            let v00 = idx(i, j);
            let v10 = idx(i + 1, j);
            let v01 = idx(i, j + 1);
            let v11 = idx(i + 1, j + 1);

            // triangle 1
            t0.push(v00);
            t1.push(v10);
            t2.push(v11);

            // triangle 2
            t0.push(v00);
            t1.push(v11);
            t2.push(v01);
        }
    }

    TriMesh {
        vx: Box::leak(vx.into_boxed_slice()),
        vy: Box::leak(vy.into_boxed_slice()),
        t0: Box::leak(t0.into_boxed_slice()),
        t1: Box::leak(t1.into_boxed_slice()),
        t2: Box::leak(t2.into_boxed_slice()),
    }
}

pub fn make_single_cube_5tet_mesh() -> TetMesh<'static> {
    let vx = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let vy = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let vz = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    // 5-tet decomposition of cube
    let t0 = vec![0, 1, 3, 4, 6];
    let t1 = vec![1, 2, 3, 6, 5];
    let t2 = vec![1, 3, 4, 6, 5];
    let t2b = vec![3, 4, 6, 7, 5]; // optional variant

    let t0 = vec![0, 1, 3, 4, 6];
    let t1 = vec![1, 2, 3, 6, 5];
    let t2 = vec![1, 3, 4, 6, 5];
    let t3 = vec![3, 4, 6, 7, 5];
    let t4 = vec![1, 3, 5, 6, 4];

    TetMesh {
        vx: Box::leak(vx.into_boxed_slice()),
        vy: Box::leak(vy.into_boxed_slice()),
        vz: Box::leak(vz.into_boxed_slice()),
        t0: Box::leak(t0.into_boxed_slice()),
        t1: Box::leak(t1.into_boxed_slice()),
        t2: Box::leak(t2.into_boxed_slice()),
        t3: Box::leak(t3.into_boxed_slice()),
    }
}

pub fn make_star_tet_mesh(n: usize) -> TetMesh<'static> {
    assert!(n >= 4);

    use std::f64::consts::PI;

    let mut vx = vec![0.0];
    let mut vy = vec![0.0];
    let mut vz = vec![0.0];

    // Points on sphere
    for i in 0..(n + 2) {
        let theta = 2.0 * PI * (i as f64) / (n as f64);
        let phi = PI * ((i as f64) / (n as f64));

        vx.push(theta.cos() * phi.sin());
        vy.push(theta.sin() * phi.sin());
        vz.push(phi.cos());
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();
    let mut t3 = Vec::new();

    for i in 1..=n {
        t0.push(0);
        t1.push(i);
        t2.push(i + 1);
        t3.push(i + 2);
    }

    TetMesh {
        vx: Box::leak(vx.into_boxed_slice()),
        vy: Box::leak(vy.into_boxed_slice()),
        vz: Box::leak(vz.into_boxed_slice()),
        t0: Box::leak(t0.into_boxed_slice()),
        t1: Box::leak(t1.into_boxed_slice()),
        t2: Box::leak(t2.into_boxed_slice()),
        t3: Box::leak(t3.into_boxed_slice()),
    }
}
