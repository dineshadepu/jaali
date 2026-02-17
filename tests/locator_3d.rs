mod utils;
use jaali::{Backend, Locator3D, TetMesh};
use rand::Rng;
use utils::*;
use utils::{
    assert_locator3d_backends_agree, assert_single_query_expected_count,
    make_large_unstructured_tet_mesh,
};

// ------------------------------------------------------------
// Basic correctness test (strictly inside)
// ------------------------------------------------------------
#[test]
fn locator3d_basic_inside_all_backends() {
    // Single tetrahedron
    let mesh = single_tet();

    let qx = vec![0.1];
    let qy = vec![0.1];
    let qz = vec![0.1];

    for backend in backends() {
        let mut out = vec![-1];
        let mut locator = jaali::Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");
        locator.locate(&qx, &qy, &qz, &mut out);
        assert_eq!(out[0], 0, "Locator3D failed for backend {:?}", backend);
    }
}

// ------------------------------------------------------------
// Outside-of-mesh correctness (3D)
// ------------------------------------------------------------
#[test]
fn locator3d_basic_outside_all_backends() {
    let mesh = make_single_cube_5tet_mesh(); // or any simple tet mesh

    // Clearly outside
    let qx = vec![1.5, -0.5, 0.5];
    let qy = vec![1.5, 0.5, -0.5];
    let qz = vec![1.5, 0.5, 0.5];

    for backend in available_backends() {
        let mut out = vec![-2; qx.len()]; // sentinel
        let mut locator = Locator3D::new(&mesh).with_backend(backend).unwrap();

        locator.locate(&qx, &qy, &qz, &mut out);

        for (i, &id) in out.iter().enumerate() {
            assert_eq!(
                id, -1,
                "Expected outside point to return -1 (backend {:?}, q={})",
                backend, i
            );
        }
    }
}

#[test]
fn locator3d_vertex_shared_by_20_tets() {
    let mut vx = vec![0.0];
    let mut vy = vec![0.0];
    let mut vz = vec![0.0];

    // 20 points roughly on a sphere
    for i in 0..20 {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / 20.0;
        vx.push(theta.cos());
        vy.push(theta.sin());
        vz.push(0.2 * (i as f64).sin());
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();
    let mut t3 = Vec::new();

    for i in 1..20 {
        t0.push(0);
        t1.push(i);
        t2.push(i + 1);
        t3.push(0);
    }
    // close the fan
    t0.push(0);
    t1.push(20);
    t2.push(1);
    t3.push(0);

    let mesh = TetMesh {
        vx: &vx,
        vy: &vy,
        vz: &vz,
        t0: &t0,
        t1: &t1,
        t2: &t2,
        t3: &t3,
    };

    let qx = vec![0.0];
    let qy = vec![0.0];
    let qz = vec![0.0];

    assert_single_query_expected_count(&mesh, [qx[0], qy[0], qz[0]], 20);
    assert_locator3d_backends_agree(&mesh, &qx, &qy, &qz);
}

#[test]
fn stress_random_queries_large_batch() {
    use rand::Rng;

    let mesh = make_large_unstructured_tet_mesh(); // thousands of tets

    let n = 100_000;
    let mut rng = rand::thread_rng();

    let qx: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.2..1.2)).collect();
    let qy: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.2..1.2)).collect();
    let qz: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.2..1.2)).collect();

    assert_locator3d_backends_agree(&mesh, &qx, &qy, &qz);
}

#[test]
fn stress_boundary_points() {
    let mesh = make_large_unstructured_tet_mesh();

    let mut qx = Vec::new();
    let mut qy = Vec::new();
    let mut qz = Vec::new();

    // hammer faces z = 0.5
    for i in 0..50_000 {
        let t = i as f64 / 50_000.0;
        qx.push(t);
        qy.push(0.5);
        qz.push(0.5);
    }

    assert_locator3d_backends_agree(&mesh, &qx, &qy, &qz);
}

#[cfg(feature = "gpu")]
#[test]
fn stress_repeated_calls_same_locator_all_backends() {
    use rand::Rng;
    use std::collections::HashSet;

    let mesh = make_large_unstructured_tet_mesh();
    let mut rng = rand::thread_rng();

    let n = 10_000;
    let max_hits = 32;

    // Persistent locators (one per backend)
    let mut loc_serial = Locator3D::new(&mesh).with_backend(Backend::Serial).unwrap();

    let mut loc_parallel = Locator3D::new(&mesh)
        .with_backend(Backend::ParallelCPU)
        .unwrap();

    let mut loc_gpu = Locator3D::new(&mesh).with_backend(Backend::GPU).unwrap();

    for iter in 0..50 {
        // Same queries for all backends
        let qx: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
        let qy: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
        let qz: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

        loc_serial.locate_all(&qx, &qy, &qz).unwrap();
        loc_parallel.locate_all(&qx, &qy, &qz).unwrap();
        loc_gpu.locate_all(&qx, &qy, &qz).unwrap();

        // -------- compare counts --------
        assert_eq!(
            loc_serial.counts, loc_parallel.counts,
            "Counts differ: serial vs parallel (iter {})",
            iter
        );

        assert_eq!(
            loc_serial.counts, loc_gpu.counts,
            "Counts differ: serial vs GPU (iter {})",
            iter
        );

        // -------- compare hit sets per query --------
        for q in 0..n {
            let c = loc_serial.counts[q] as usize;
            assert!(c <= max_hits);

            let base = q * max_hits;

            let ids_serial: HashSet<i32> =
                loc_serial.indices[base..base + c].iter().copied().collect();

            let ids_parallel: HashSet<i32> = loc_parallel.indices[base..base + c]
                .iter()
                .copied()
                .collect();

            let ids_gpu: HashSet<i32> = loc_gpu.indices[base..base + c].iter().copied().collect();

            assert_eq!(
                ids_serial, ids_parallel,
                "ID mismatch serial vs parallel at q={}, iter={}",
                q, iter
            );

            assert_eq!(
                ids_serial, ids_gpu,
                "ID mismatch serial vs GPU at q={}, iter={}",
                q, iter
            );
        }
    }
}

#[test]
fn stress_variable_batch_sizes_same_locator() {
    let mesh = make_large_unstructured_tet_mesh();

    let mut locator = Locator3D::new(&mesh).with_backend(Backend::Serial).unwrap();

    for &n in &[1, 7, 64, 513, 4096, 10_000, 128, 3] {
        let qx = vec![0.3; n];
        let qy = vec![0.3; n];
        let qz = vec![0.3; n];

        locator.locate_all(&qx, &qy, &qz).unwrap();

        // sanity checks
        assert!(locator.indices.len() >= n * locator.max_hits);
        assert!(locator.counts.len() >= n);

        for q in 0..n {
            let c = locator.counts[q] as usize;
            assert!(c <= locator.max_hits);

            let base = q * locator.max_hits;
            for i in 0..c {
                assert!(locator.indices[base + i] >= 0);
            }
        }
    }
}

// #[cfg(feature = "gpu")]
// #[test]
// fn stress_variable_batch_sizes_same_locator_gpu() {
//     let mesh = make_large_unstructured_tet_mesh();

//     let mut locator = Locator3D::new(&mesh).with_backend(Backend::GPU).unwrap();

//     for &n in &[1, 7, 64, 513, 4096, 10_000, 256, 5] {
//         let qx = vec![0.3; n];
//         let qy = vec![0.3; n];
//         let qz = vec![0.3; n];

//         locator.locate_all(&qx, &qy, &qz).unwrap();

//         for &c in &locator.counts[..n] {
//             assert!((c as usize) <= locator.max_hits);
//         }
//     }
// }

#[cfg(feature = "gpu")]
#[test]
fn stress_variable_batch_sizes_same_locator_all_backends() {
    use std::collections::HashSet;

    let mesh = make_large_unstructured_tet_mesh();
    let max_hits = 32;

    // Persistent locators (THIS is what we are testing)
    let mut loc_serial = Locator3D::new_with_capacity(&mesh, 1, max_hits)
        .with_backend(Backend::Serial)
        .unwrap();

    let mut loc_parallel = Locator3D::new_with_capacity(&mesh, 1, max_hits)
        .with_backend(Backend::ParallelCPU)
        .unwrap();

    let mut loc_gpu = Locator3D::new_with_capacity(&mesh, 1, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    // Deliberately non-monotonic sizes
    for &n in &[1, 7, 64, 513, 4096, 10_000, 256, 5] {
        let qx = vec![0.3; n];
        let qy = vec![0.3; n];
        let qz = vec![0.3; n];

        // Run all backends on identical queries
        loc_serial.locate_all(&qx, &qy, &qz).unwrap();
        loc_parallel.locate_all(&qx, &qy, &qz).unwrap();
        loc_gpu.locate_all(&qx, &qy, &qz).unwrap();

        // ---------- counts must match ----------
        assert_eq!(
            &loc_serial.counts[..n],
            &loc_parallel.counts[..n],
            "Serial vs Parallel counts mismatch (n={})",
            n
        );

        assert_eq!(
            &loc_serial.counts[..n],
            &loc_gpu.counts[..n],
            "Serial vs GPU counts mismatch (n={})",
            n
        );

        // ---------- hit sets must match per query ----------
        for q in 0..n {
            let c = loc_serial.counts[q] as usize;
            assert!(c <= max_hits);

            let base = q * max_hits;

            let ids_serial: HashSet<i32> =
                loc_serial.indices[base..base + c].iter().copied().collect();

            let ids_parallel: HashSet<i32> = loc_parallel.indices[base..base + c]
                .iter()
                .copied()
                .collect();

            let ids_gpu: HashSet<i32> = loc_gpu.indices[base..base + c].iter().copied().collect();

            assert_eq!(
                ids_serial, ids_parallel,
                "Serial vs Parallel ID mismatch at q={}, n={}",
                q, n
            );

            assert_eq!(
                ids_serial, ids_gpu,
                "Serial vs GPU ID mismatch at q={}, n={}",
                q, n
            );
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn stress_3d_idempotent_repeated_calls() {
    let mesh = make_large_unstructured_tet_mesh();
    let n = 10_000;
    let max_hits = 32;

    let qx = vec![0.33; n];
    let qy = vec![0.44; n];
    let qz = vec![0.55; n];

    let mut loc = Locator3D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    loc.locate_all(&qx, &qy, &qz).unwrap();
    let counts_ref = loc.counts.clone();
    let indices_ref = loc.indices.clone();

    for _ in 0..20 {
        loc.locate_all(&qx, &qy, &qz).unwrap();
        assert_eq!(loc.counts, counts_ref);
        assert_eq!(loc.indices, indices_ref);
    }
}

#[test]
fn stress_3d_high_valence_vertex_32_tets() {
    let mesh = make_star_tet_mesh(32); // construct programmatically
    let qx = vec![0.0];
    let qy = vec![0.0];
    let qz = vec![0.0];

    for backend in backends() {
        let mut loc = Locator3D::new_with_capacity(&mesh, 1, 64)
            .with_backend(backend)
            .unwrap();

        loc.locate_all(&qx, &qy, &qz).unwrap();
        assert_eq!(loc.counts[0] as usize, 32);
    }
}

#[test]
fn stress_3d_epsilon_sweep_near_face() {
    let mesh = make_large_unstructured_tet_mesh();

    let base = 0.5;
    for backend in backends() {
        let mut loc = Locator3D::new(&mesh).with_backend(backend).unwrap();

        for k in -10..=10 {
            let eps = (k as f64) * 1e-13;
            let qx = vec![base];
            let qy = vec![base];
            let qz = vec![base + eps];

            loc.locate_all(&qx, &qy, &qz).unwrap();
            assert!(loc.counts[0] >= 1);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn stress_3d_many_queries_tiny_mesh() {
    let mesh = make_single_cube_5tet_mesh();
    let n = 200_000;
    let max_hits = 8;

    let qx = vec![0.25; n];
    let qy = vec![0.25; n];
    let qz = vec![0.25; n];

    let mut loc = Locator3D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    loc.locate_all(&qx, &qy, &qz).unwrap();

    for &c in &loc.counts {
        assert!(
            (c as usize) <= max_hits,
            "counts must never exceed max_hits"
        );
    }
}

#[test]
fn sanity_star_tet_mesh_counts() {
    let n = 32;
    let mesh = make_star_tet_mesh(n);

    let qx = vec![0.0];
    let qy = vec![0.0];
    let qz = vec![0.0];

    let mut loc = Locator3D::new_with_capacity(&mesh, 1, 64)
        .with_backend(Backend::Serial)
        .unwrap();

    loc.locate_all(&qx, &qy, &qz).unwrap();
    assert_eq!(loc.counts[0] as usize, n);
}

// =================================================
// Testing the locate method
// =================================================
#[test]
fn locate_matches_locate_all_min_id() {
    let mesh = make_large_unstructured_tet_mesh();

    let qx = vec![0.3, 0.6, 0.9];
    let qy = vec![0.3, 0.6, 0.9];
    let qz = vec![0.3, 0.6, 0.9];

    for backend in available_backends() {
        let mut locator = Locator3D::new(&mesh).with_backend(backend).unwrap();

        let mut out = vec![-1; qx.len()];
        locator.locate(&qx, &qy, &qz, &mut out);

        locator.locate_all(&qx, &qy, &qz).unwrap();

        for q in 0..qx.len() {
            let expected = reference_locate_from_locate_all(
                &locator.indices,
                &locator.counts,
                locator.max_hits,
                q,
            );
            assert_eq!(
                out[q], expected,
                "locate != min-id locate_all on backend {:?}",
                backend
            );
        }
    }
}

#[test]
fn locate_agrees_across_backends() {
    let mesh = make_large_unstructured_tet_mesh();

    let n = 10_000;
    let qx = vec![0.37; n];
    let qy = vec![0.41; n];
    let qz = vec![0.53; n];

    assert_locate_agrees_across_backends(&mesh, &qx, &qy, &qz);
}

#[test]
fn locate_idempotent_repeated_calls_same_locator_all_backends() {
    use rand::Rng;

    let mesh = make_large_unstructured_tet_mesh();
    let mut rng = rand::thread_rng();

    let n = 5000;

    for backend in available_backends() {
        let mut locator = Locator3D::new(&mesh).with_backend(backend).unwrap();

        let qx: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
        let qy: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
        let qz: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

        let mut ref_out = vec![-1; n];
        locator.locate(&qx, &qy, &qz, &mut ref_out);

        for _ in 0..10 {
            let mut out = vec![-1; n];
            locator.locate(&qx, &qy, &qz, &mut out);
            assert_eq!(
                ref_out, out,
                "non-idempotent locate on backend {:?}",
                backend
            );
        }
    }
}

#[test]
fn locate_variable_batch_sizes_same_locator_all_backends() {
    let mesh = make_large_unstructured_tet_mesh();

    for backend in available_backends() {
        let mut locator = Locator3D::new(&mesh).with_backend(backend).unwrap();

        for &n in &[1, 7, 64, 513, 4096, 128, 3] {
            let qx = vec![0.25; n];
            let qy = vec![0.25; n];
            let qz = vec![0.25; n];

            let mut out = vec![-1; n];
            locator.locate(&qx, &qy, &qz, &mut out);

            for &id in &out {
                assert!(id >= -1);
            }
        }
    }
}

#[test]
fn locate_high_valence_vertex_min_id_all_backends() {
    let mesh = make_star_tet_mesh(32);

    let qx = vec![0.0];
    let qy = vec![0.0];
    let qz = vec![0.0];

    let mut ref_out = None;

    for backend in available_backends() {
        let mut out = vec![-1];

        Locator3D::new_with_capacity(&mesh, 1, 64)
            .with_backend(backend)
            .unwrap()
            .locate(&qx, &qy, &qz, &mut out);

        assert_eq!(out[0], 0, "wrong owner on backend {:?}", backend);

        if let Some(ref expected) = ref_out {
            assert_eq!(
                out, *expected,
                "backend {:?} disagrees with others",
                backend
            );
        } else {
            ref_out = Some(out);
        }
    }
}
