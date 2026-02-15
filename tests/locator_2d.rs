mod utils;
use utils::*;

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
        let locator = jaali::Locator2D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");
        locator.locate(&qx, &qy, &mut out);
        assert_eq!(out[0], 0, "Locator2D failed for backend {:?}", backend);
    }
}

// ------------------------------------------------------------
// Stress test (BVH vs brute force)
// ------------------------------------------------------------
#[test]
#[ignore]
fn stress_locator2d_vs_bruteforce() {
    let (vx, vy, t0, t1, t2) = generate_grid_mesh_2d(200, 200);

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    // Strictly inside points
    let n = 10_000;
    let mut qx = Vec::with_capacity(n);
    let mut qy = Vec::with_capacity(n);

    for i in 0..n {
        qx.push((i % 199) as f64 + 0.3);
        qy.push(((i / 199) % 199) as f64 + 0.3);
    }

    let brute: Vec<i32> = qx
        .iter()
        .zip(&qy)
        .map(|(&x, &y)| brute_force_find_2d(x, y, &mesh))
        .collect();

    for backend in backends() {
        let locator = Locator2D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        let mut out = vec![-1; n];
        locator.locate(&qx, &qy, &mut out);

        let mismatches = out
            .iter()
            .zip(&brute)
            .filter(|(a, b)| (**a >= 0) != (**b >= 0))
            .count();

        assert_eq!(mismatches, 0, "Mismatch for backend {:?}", backend);
    }
}
