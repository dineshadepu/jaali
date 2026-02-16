mod utils;
use jaali::{Backend, LocateMode, Locator3D, TetMesh};
use rand::Rng;
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
        let mut locator = jaali::Locator3D::new_with_capacity(&mesh, qx.len(), 16)
            .with_backend(backend)
            .expect("backend init failed");
        locator.locate(&qx, &qy, &qz, &mut out);
        assert_eq!(out[0], 0, "Locator3D failed for backend {:?}", backend);
    }
}

// // ------------------------------------------------------------
// // Stress test (BVH vs brute force)
// // ------------------------------------------------------------
// #[test]
// #[ignore]
// fn stress_locator3d_vs_bruteforce() {
//     // Small grid of tets (kept reasonable for CI)
//     let vx = vec![0.0, 1.0, 0.0, 0.0];
//     let vy = vec![0.0, 0.0, 1.0, 0.0];
//     let vz = vec![0.0, 0.0, 0.0, 1.0];

//     let t0 = vec![0usize];
//     let t1 = vec![1usize];
//     let t2 = vec![2usize];
//     let t3 = vec![3usize];

//     let mesh = TetMesh {
//         vx: &vx,
//         vy: &vy,
//         vz: &vz,
//         t0: &t0,
//         t1: &t1,
//         t2: &t2,
//         t3: &t3,
//     };

//     let qx = vec![0.1; 10_000];
//     let qy = vec![0.1; 10_000];
//     let qz = vec![0.1; 10_000];

//     let brute: Vec<i32> = qx
//         .iter()
//         .zip(&qy)
//         .zip(&qz)
//         .map(|((&x, &y), &z)| brute_force_find_3d(x, y, z, &mesh))
//         .collect();

//     for backend in backends() {
//         let locator = Locator3D::new(&mesh)
//             .with_backend(backend)
//             .expect("backend init failed");

//         let mut out = vec![-1; qx.len()];
//         locator.locate(&qx, &qy, &qz, &mut out);

//         let mismatches = out
//             .iter()
//             .zip(&brute)
//             .filter(|(a, b)| (**a >= 0) != (**b >= 0))
//             .count();

//         assert_eq!(mismatches, 0, "Mismatch for backend {:?}", backend);
//     }
// }

// #[test]
// fn locator3d_large_mesh_boundary_cases() {
//     let (vx, vy, vz, t0, t1, t2, t3) = grid_mesh_3d(20, 20, 20);
//     let mesh = TetMesh {
//         vx: &vx,
//         vy: &vy,
//         vz: &vz,
//         t0: &t0,
//         t1: &t1,
//         t2: &t2,
//         t3: &t3,
//     };
//     let locator = Locator3D::new(&mesh);

//     let eps = 1e-12;

//     let qx = vec![
//         5.2,       // interior
//         5.0,       // face
//         5.0 + eps, // near-face inside
//         5.0 - eps, // near-face outside
//         5.0,       // edge
//         5.0,       // vertex
//         100.0,     // outside
//     ];

//     let qy = vec![5.2, 5.2, 5.2, 5.2, 5.0, 5.0, 100.0];
//     let qz = vec![5.2, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0];

//     let mut out = vec![-1; qx.len()];

//     locator
//         .locate_with_mode(&qx, &qy, &qz, &mut out, LocateMode::StrictInside)
//         .unwrap();
//     assert!(out[0] >= 0);
//     assert_eq!(out[1], -1);
//     assert!(out[2] >= 0);
//     assert_eq!(out[3], -1);
//     assert_eq!(out[5], -1);

//     locator
//         .locate_with_mode(&qx, &qy, &qz, &mut out, LocateMode::InsideOrBoundary)
//         .unwrap();
//     assert!(out[1] >= 0);
//     assert!(out[4] >= 0);
//     assert!(out[5] >= 0);
// }

// #[test]
// fn locator3d_random_near_boundary_fuzz() {
//     let (vx, vy, vz, t0, t1, t2, t3) = grid_mesh_3d(20, 20, 20);
//     let mesh = TetMesh {
//         vx: &vx,
//         vy: &vy,
//         vz: &vz,
//         t0: &t0,
//         t1: &t1,
//         t2: &t2,
//         t3: &t3,
//     };

//     let locator = Locator3D::new(&mesh);

//     let mut rng = rand::thread_rng();
//     let eps = 1e-10;

//     let mut qx = Vec::new();
//     let mut qy = Vec::new();
//     let mut qz = Vec::new();

//     for _ in 0..200 {
//         let x = rng.gen_range(1.0..18.0);
//         qx.push(x + eps * rng.gen_range(-1.0..1.0));
//         qy.push(x);
//         qz.push(x);
//     }

//     let mut out = vec![-1; qx.len()];
//     locator
//         .locate_with_mode(&qx, &qy, &qz, &mut out, LocateMode::InsideOrBoundary)
//         .unwrap();

//     assert!(out.iter().all(|&v| v >= 0));
// }
