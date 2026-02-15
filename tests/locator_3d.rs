mod utils;
use jaali::{Backend, Locator3D, TetMesh};
use utils::*;

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
        let locator = jaali::Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");
        locator.locate(&qx, &qy, &qz, &mut out);
        assert_eq!(out[0], 0, "Locator3D failed for backend {:?}", backend);
    }
}

// ------------------------------------------------------------
// Stress test (BVH vs brute force)
// ------------------------------------------------------------
#[test]
#[ignore]
fn stress_locator3d_vs_bruteforce() {
    // Small grid of tets (kept reasonable for CI)
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

    let qx = vec![0.1; 10_000];
    let qy = vec![0.1; 10_000];
    let qz = vec![0.1; 10_000];

    let brute: Vec<i32> = qx
        .iter()
        .zip(&qy)
        .zip(&qz)
        .map(|((&x, &y), &z)| brute_force_find_3d(x, y, z, &mesh))
        .collect();

    for backend in backends() {
        let locator = Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        let mut out = vec![-1; qx.len()];
        locator.locate(&qx, &qy, &qz, &mut out);

        let mismatches = out
            .iter()
            .zip(&brute)
            .filter(|(a, b)| (**a >= 0) != (**b >= 0))
            .count();

        assert_eq!(mismatches, 0, "Mismatch for backend {:?}", backend);
    }
}
