mod utils;
use utils::*;
use utils::{collect_hits_2d, make_large_unstructured_tri_mesh_2d};

use std::collections::HashSet;

use jaali::{Backend, Locator2D, TriMesh};

// ------------------------------------------------------------
// Basic correctness test (strictly inside)
// ------------------------------------------------------------
#[test]
fn locator2d_basic_inside_all_backends() {
    // Single triangle
    let mesh = single_triangle();

    // Strictly inside
    let qx = vec![0.25];
    let qy = vec![0.25];

    for backend in backends() {
        let mut out = vec![-1];
        let mut locator = jaali::Locator2D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");
        locator.locate(&qx, &qy, &mut out);
        assert_eq!(out[0], 0, "Locator2D failed for backend {:?}", backend);
    }
}

// ------------------------------------------------------------
// Outside-of-mesh correctness (2D)
// ------------------------------------------------------------
#[test]
fn locator2d_basic_outside_all_backends() {
    let mesh = single_triangle();

    // Clearly outside
    let qx = vec![1.5, -0.5, 0.5];
    let qy = vec![1.5, 0.5, -0.5];

    for backend in backends() {
        let mut out = vec![-2; qx.len()]; // sentinel
        let mut locator = jaali::Locator2D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        locator.locate(&qx, &qy, &mut out);

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
fn locator2d_boundary_cases_all_backends() {
    let mesh = single_triangle();

    let qx = vec![
        0.25, // inside
        0.5,  // edge
        0.0,  // vertex
        1.5,  // outside
    ];
    let qy = vec![0.25, 0.0, 0.0, 1.5];

    let n = qx.len();
    let max_hits = 4;

    for backend in backends() {
        // ---------- StrictInside ----------
        let mut locator = Locator2D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        locator.locate_all(&qx, &qy).unwrap();

        let counts = &locator.counts;
        let indices = &locator.indices;

        // inside
        assert_eq!(counts[0], 1);
        assert_eq!(indices[0 * max_hits], 0);

        // edge
        assert_eq!(counts[1], 1);

        // vertex
        assert_eq!(counts[2], 1);

        // outside
        assert_eq!(counts[3], 0);
    }
}

#[test]
fn locator2d_large_mesh_boundary_cases() {
    let (vx, vy, t0, t1, t2) = generate_grid_mesh_2d(50, 50);

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    let eps = 1e-12;

    let qx = vec![
        10.5,       // interior
        10.0,       // vertical grid line (edge)
        10.0 + eps, // near-edge inside
        10.0 - eps, // near-edge outside
        10.0,       // vertex
        100.0,      // outside
    ];

    let qy = vec![10.5, 10.5, 10.5, 10.5, 10.0, 100.0];

    let n = qx.len();
    let H = 8;

    for backend in backends() {
        let mut locator = Locator2D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        locator.locate_all(&qx, &qy).unwrap();

        let counts = &locator.counts;

        // interior
        assert!(counts[0] >= 1, "interior point should hit");

        // edge (shared by two triangles in grid)
        assert!(counts[1] >= 1, "edge point should hit");

        // epsilon inside
        assert!(counts[2] >= 1, "epsilon-inside should hit");

        // epsilon near-edge: may belong to neighbor cell
        assert!(counts[3] >= 1, "near-edge point should hit some cell");

        // vertex (shared by 4 triangles in grid interior)
        assert!(counts[4] >= 1, "vertex point should hit");

        // outside
        assert_eq!(counts[5], 0, "outside point should not hit");
    }
}

#[test]
fn locator2d_center_shared_by_8_triangles() {
    // ------------------------
    // Mesh: square split into 8 triangles
    // ------------------------
    let vx = vec![
        0.0, // 0
        1.0, // 1
        1.0, // 2
        0.0, // 3
        0.5, // 4 (center)
    ];

    let vy = vec![
        0.0, // 0
        0.0, // 1
        1.0, // 2
        1.0, // 3
        0.5, // 4 (center)
    ];

    // 8 triangles sharing the center
    let t0 = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let t1 = vec![1, 2, 3, 0, 4, 4, 4, 4];
    let t2 = vec![4, 4, 4, 4, 1, 2, 3, 0];

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    // ------------------------
    // Query: center point
    // ------------------------
    let qx = vec![0.5];
    let qy = vec![0.5];

    // Allow up to 8 hits
    let mut locator = Locator2D::new(&mesh);

    locator.locate_all(&qx, &qy).unwrap();

    // ------------------------
    // Assertions
    // ------------------------
    let count = locator.counts[0] as usize;
    assert_eq!(
        count, 8,
        "Center point should belong to exactly 8 triangles"
    );

    // Optional sanity check: all IDs valid
    let base = 0 * locator.max_hits;
    for i in 0..count {
        let tid = locator.indices[base + i];
        assert!(
            tid >= 0 && (tid as usize) < t0.len(),
            "Invalid triangle id {}",
            tid
        );
    }
}

#[test]
fn stress_2d_random_queries_large_batch() {
    use rand::Rng;

    let mesh = make_large_unstructured_tri_mesh_2d(200, 200);
    let n = 100_000;
    let max_hits = 8;

    let mut rng = rand::thread_rng();
    let qx: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..210.0)).collect();
    let qy: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..210.0)).collect();

    let mut results = Vec::new();

    for backend in backends() {
        let mut loc = Locator2D::new_with_capacity(&mesh, n, max_hits)
            .with_backend(backend)
            .unwrap();

        loc.locate_all(&qx, &qy).unwrap();

        results.push((
            loc.counts.clone(),
            collect_hits_2d(&loc.indices, &loc.counts, max_hits, n),
        ));
    }

    // compare all backends against the first
    for i in 1..results.len() {
        assert_eq!(results[0].0, results[i].0, "counts mismatch");
        assert_eq!(results[0].1, results[i].1, "hit sets mismatch");
    }
}

#[test]
fn stress_2d_boundary_points() {
    let mesh = make_large_unstructured_tri_mesh_2d(200, 200);
    let max_hits = 8;

    let mut qx = Vec::new();
    let mut qy = Vec::new();

    // vertical grid line x = 50
    for i in 0..50_000 {
        let t = i as f64 / 50_000.0 * 200.0;
        qx.push(50.0);
        qy.push(t);
    }

    let n = qx.len();
    let mut results = Vec::new();

    for backend in backends() {
        let mut loc = Locator2D::new_with_capacity(&mesh, n, max_hits)
            .with_backend(backend)
            .unwrap();

        loc.locate_all(&qx, &qy).unwrap();

        results.push((
            loc.counts.clone(),
            collect_hits_2d(&loc.indices, &loc.counts, max_hits, n),
        ));
    }

    for i in 1..results.len() {
        assert_eq!(results[0].0, results[i].0);
        assert_eq!(results[0].1, results[i].1);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn stress_2d_repeated_calls_same_locator_all_backends() {
    use rand::Rng;

    let mesh = make_large_unstructured_tri_mesh_2d(100, 100);
    let n = 10_000;
    let max_hits = 8;

    let mut rng = rand::thread_rng();

    let mut loc_serial = Locator2D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::Serial)
        .unwrap();

    let mut loc_parallel = Locator2D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::ParallelCPU)
        .unwrap();

    let mut loc_gpu = Locator2D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    for iter in 0..50 {
        let qx: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();
        let qy: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();

        loc_serial.locate_all(&qx, &qy).unwrap();
        loc_parallel.locate_all(&qx, &qy).unwrap();
        loc_gpu.locate_all(&qx, &qy).unwrap();

        assert_eq!(
            loc_serial.counts, loc_parallel.counts,
            "serial vs parallel counts mismatch at iter {}",
            iter
        );
        assert_eq!(
            loc_serial.counts, loc_gpu.counts,
            "serial vs gpu counts mismatch at iter {}",
            iter
        );

        for q in 0..n {
            let c = loc_serial.counts[q] as usize;
            let base = q * max_hits;

            let s: HashSet<i32> = loc_serial.indices[base..base + c].iter().copied().collect();
            let p: HashSet<i32> = loc_parallel.indices[base..base + c]
                .iter()
                .copied()
                .collect();
            let g: HashSet<i32> = loc_gpu.indices[base..base + c].iter().copied().collect();

            assert_eq!(
                s, p,
                "serial vs parallel mismatch at q={}, iter={}",
                q, iter
            );
            assert_eq!(s, g, "serial vs gpu mismatch at q={}, iter={}", q, iter);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn stress_2d_variable_batch_sizes_same_locator_all_backends() {
    let mesh = make_large_unstructured_tri_mesh_2d(100, 100);
    let max_hits = 8;

    let mut loc_serial = Locator2D::new_with_capacity(&mesh, 1, max_hits)
        .with_backend(Backend::Serial)
        .unwrap();

    let mut loc_parallel = Locator2D::new_with_capacity(&mesh, 1, max_hits)
        .with_backend(Backend::ParallelCPU)
        .unwrap();

    let mut loc_gpu = Locator2D::new_with_capacity(&mesh, 1, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    for &n in &[1, 7, 64, 513, 4096, 10_000, 256, 5] {
        let qx = vec![0.3; n];
        let qy = vec![0.3; n];

        loc_serial.locate_all(&qx, &qy).unwrap();
        loc_parallel.locate_all(&qx, &qy).unwrap();
        loc_gpu.locate_all(&qx, &qy).unwrap();

        assert_eq!(&loc_serial.counts[..n], &loc_parallel.counts[..n]);
        assert_eq!(&loc_serial.counts[..n], &loc_gpu.counts[..n]);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn stress_2d_idempotent_repeated_calls() {
    let mesh = make_large_unstructured_tri_mesh_2d(100, 100);
    let n = 10_000;
    let max_hits = 8;

    let qx = vec![42.42; n];
    let qy = vec![17.17; n];

    let mut loc = Locator2D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    loc.locate_all(&qx, &qy).unwrap();
    let counts_ref = loc.counts.clone();
    let indices_ref = loc.indices.clone();

    for _ in 0..20 {
        loc.locate_all(&qx, &qy).unwrap();
        assert_eq!(loc.counts, counts_ref);
        assert_eq!(loc.indices, indices_ref);
    }
}

#[test]
fn stress_2d_high_valence_vertex() {
    let n = 32;
    let mut vx = vec![0.0];
    let mut vy = vec![0.0];

    for i in 0..n {
        let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        vx.push(t.cos());
        vy.push(t.sin());
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();

    for i in 1..n {
        t0.push(0);
        t1.push(i);
        t2.push(i + 1);
    }
    t0.push(0);
    t1.push(n);
    t2.push(1);

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    let qx = vec![0.0];
    let qy = vec![0.0];

    for backend in backends() {
        let mut loc = Locator2D::new_with_capacity(&mesh, 1, 32)
            .with_backend(backend)
            .unwrap();

        loc.locate_all(&qx, &qy).unwrap();
        assert_eq!(loc.counts[0] as usize, n);
    }
}

#[test]
fn stress_2d_epsilon_sweep_near_edge() {
    let mesh = make_large_unstructured_tri_mesh_2d(10, 10);

    let base_x = 5.0;
    let y = 5.5;

    for backend in backends() {
        let mut loc = Locator2D::new(&mesh).with_backend(backend).unwrap();

        for k in -10..=10 {
            let eps = (k as f64) * 1e-14;
            let qx = vec![base_x + eps];
            let qy = vec![y];

            loc.locate_all(&qx, &qy).unwrap();
            assert!(loc.counts[0] >= 1);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn stress_2d_many_queries_tiny_mesh() {
    let mesh = make_large_unstructured_tri_mesh_2d(1, 1);
    let n = 200_000;
    let max_hits = 4;

    let qx = vec![0.25; n];
    let qy = vec![0.25; n];

    let mut loc = Locator2D::new_with_capacity(&mesh, n, max_hits)
        .with_backend(Backend::GPU)
        .unwrap();

    loc.locate_all(&qx, &qy).unwrap();

    for &c in &loc.counts {
        assert!(c >= 1, "each query must hit at least one triangle");
        assert!((c as usize) <= max_hits, "counts must not exceed max_hits");
    }
}

// =================================================
// Testing the locate method
// =================================================
#[test]
fn locate_2d_matches_locate_all_min_id() {
    let mesh = make_large_unstructured_tri_mesh_2d(20, 20);

    let qx = vec![3.3, 7.7, 15.5];
    let qy = vec![3.3, 7.7, 15.5];

    for backend in available_backends() {
        let mut loc = Locator2D::new(&mesh).with_backend(backend).unwrap();

        let mut out = vec![-1; qx.len()];
        loc.locate(&qx, &qy, &mut out);

        loc.locate_all(&qx, &qy).unwrap();

        for q in 0..qx.len() {
            let expected =
                reference_locate_2d_from_locate_all(&loc.indices, &loc.counts, loc.max_hits, q);
            assert_eq!(
                out[q], expected,
                "2D locate != min-id locate_all on backend {:?}",
                backend
            );
        }
    }
}

#[test]
fn locate_2d_agrees_across_backends() {
    let mesh = make_large_unstructured_tri_mesh_2d(30, 30);

    let n = 20_000;
    let qx = vec![12.3; n];
    let qy = vec![9.7; n];

    let mut ref_out = None;

    for backend in available_backends() {
        let mut loc = Locator2D::new(&mesh).with_backend(backend).unwrap();

        let mut out = vec![-1; n];
        loc.locate(&qx, &qy, &mut out);

        if let Some(ref expected) = ref_out {
            assert_eq!(
                out, *expected,
                "2D locate mismatch on backend {:?}",
                backend
            );
        } else {
            ref_out = Some(out);
        }
    }
}

#[test]
fn locate_2d_idempotent_repeated_calls_same_locator_all_backends() {
    use rand::Rng;

    let mesh = make_large_unstructured_tri_mesh_2d(40, 40);
    let mut rng = rand::thread_rng();

    let n = 5000;

    for backend in available_backends() {
        let mut loc = Locator2D::new(&mesh).with_backend(backend).unwrap();

        let qx: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..40.0)).collect();
        let qy: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..40.0)).collect();

        let mut ref_out = vec![-1; n];
        loc.locate(&qx, &qy, &mut ref_out);

        for _ in 0..10 {
            let mut out = vec![-1; n];
            loc.locate(&qx, &qy, &mut out);
            assert_eq!(
                ref_out, out,
                "2D locate not idempotent on backend {:?}",
                backend
            );
        }
    }
}

#[test]
fn locate_2d_variable_batch_sizes_same_locator_all_backends() {
    let mesh = make_large_unstructured_tri_mesh_2d(25, 25);

    for backend in available_backends() {
        let mut loc = Locator2D::new(&mesh).with_backend(backend).unwrap();

        for &n in &[1, 7, 64, 513, 4096, 128, 3] {
            let qx = vec![5.5; n];
            let qy = vec![5.5; n];

            let mut out = vec![-1; n];
            loc.locate(&qx, &qy, &mut out);

            for &id in &out {
                assert!(id >= -1);
            }
        }
    }
}

#[test]
fn locate_2d_high_valence_vertex_min_id_all_backends() {
    let mesh = make_center_star_tri_mesh_2d(); // center shared by many tris

    let qx = vec![0.5];
    let qy = vec![0.5];

    let mut ref_out = None;

    for backend in available_backends() {
        let mut out = vec![-1];

        Locator2D::new_with_capacity(&mesh, 1, 16)
            .with_backend(backend)
            .unwrap()
            .locate(&qx, &qy, &mut out);

        assert_eq!(out[0], 0, "wrong owner on backend {:?}", backend);

        if let Some(ref expected) = ref_out {
            assert_eq!(
                out, *expected,
                "2D backend {:?} disagrees with others",
                backend
            );
        } else {
            ref_out = Some(out);
        }
    }
}
