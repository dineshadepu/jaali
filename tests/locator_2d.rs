mod utils;
use utils::*;

use jaali::{Backend, LocateMode, Locator2D, TriMesh};

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
        let mut locator = jaali::Locator2D::new_with_capacity(&mesh, qx.len(), 8)
            .with_backend(backend)
            .expect("backend init failed");
        locator.locate(&qx, &qy, &mut out);
        assert_eq!(out[0], 0, "Locator2D failed for backend {:?}", backend);
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
        let mut locator = Locator2D::new_with_capacity(&mesh, n, max_hits)
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
        let mut locator = Locator2D::new_with_capacity(&mesh, n, H)
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
    let mut locator = Locator2D::new_with_capacity(&mesh, 1, 8);

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

// // ------------------------------------------------------------
// // Stress test (BVH vs brute force)
// // ------------------------------------------------------------
// #[test]
// #[ignore]
// fn stress_locator2d_vs_bruteforce() {
//     let (vx, vy, t0, t1, t2) = generate_grid_mesh_2d(200, 200);

//     let mesh = TriMesh {
//         vx: &vx,
//         vy: &vy,
//         t0: &t0,
//         t1: &t1,
//         t2: &t2,
//     };

//     // Strictly inside points
//     let n = 10_000;
//     let mut qx = Vec::with_capacity(n);
//     let mut qy = Vec::with_capacity(n);

//     for i in 0..n {
//         qx.push((i % 199) as f64 + 0.3);
//         qy.push(((i / 199) % 199) as f64 + 0.3);
//     }

//     let brute: Vec<i32> = qx
//         .iter()
//         .zip(&qy)
//         .map(|(&x, &y)| brute_force_find_2d(x, y, &mesh))
//         .collect();

//     for backend in backends() {
//         let locator = Locator2D::new(&mesh)
//             .with_backend(backend)
//             .expect("backend init failed");

//         let mut out = vec![-1; n];
//         locator.locate(&qx, &qy, &mut out);

//         let mismatches = out
//             .iter()
//             .zip(&brute)
//             .filter(|(a, b)| (**a >= 0) != (**b >= 0))
//             .count();

//         assert_eq!(mismatches, 0, "Mismatch for backend {:?}", backend);
//     }
// }
