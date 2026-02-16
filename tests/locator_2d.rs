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
        // locator.locate_all(&qx, &qy);
        locator.locate(&qx, &qy, &mut out);
        assert_eq!(out[0], 0, "Locator2D failed for backend {:?}", backend);
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

// #[test]
// fn locator2d_boundary_cases() {
//     let mesh = single_triangle();

//     let pts = [
//         (0.25, 0.25), // inside
//         (0.5, 0.0),   // edge
//         (0.0, 0.0),   // vertex
//         (1.5, 1.5),   // outside
//     ];

//     let expected_strict = [0, -1, -1, -1];
//     let expected_incl = [0, 0, 0, -1];

//     for backend in backends() {
//         let locator = Locator2D::new(&mesh).with_backend(backend).unwrap();

//         let mut out = vec![-99; pts.len()];

//         locator.locate_with_mode(
//             &pts.iter().map(|p| p.0).collect::<Vec<_>>(),
//             &pts.iter().map(|p| p.1).collect::<Vec<_>>(),
//             &mut out,
//             LocateMode::StrictInside,
//         );

//         assert_eq!(out, expected_strict);

//         locator.locate_with_mode(
//             &pts.iter().map(|p| p.0).collect::<Vec<_>>(),
//             &pts.iter().map(|p| p.1).collect::<Vec<_>>(),
//             &mut out,
//             LocateMode::InsideOrBoundary,
//         );

//         assert_eq!(out, expected_incl);
//     }
// }

// // #[test]
// // fn locator2d_large_mesh_boundary_cases() {
// //     let (vx, vy, t0, t1, t2) = generate_grid_mesh_2d(50, 50);

// //     let mesh = TriMesh {
// //         vx: &vx,
// //         vy: &vy,
// //         t0: &t0,
// //         t1: &t1,
// //         t2: &t2,
// //     };

// //     for backend in backends() {
// //         let locator = Locator2D::new(&mesh).with_backend(backend).unwrap();

// //         let eps = 1e-12;

// //         let qx = vec![
// //             10.5,       // interior
// //             10.0,       // vertical grid line (edge)
// //             10.0 + eps, // near-edge inside
// //             10.0 - eps, // near-edge outside
// //             10.0,       // vertex
// //             100.0,      // outside
// //         ];

// //         let qy = vec![10.5, 10.5, 10.5, 10.5, 10.0, 100.0];

// //         let mut out = vec![-1; qx.len()];

// //         // StrictInside
// //         locator
// //             .locate_with_mode(&qx, &qy, &mut out, LocateMode::StrictInside)
// //             .unwrap();
// //         assert!(out[0] >= 0); // interior
// //         assert_eq!(out[1], -1); // edge rejected
// //         assert!(out[2] >= 0); // epsilon inside
// //         assert_eq!(out[3], -1); // epsilon outside
// //         assert_eq!(out[4], -1); // vertex rejected
// //         assert_eq!(out[5], -1); // outside

// //         // InsideOrBoundary
// //         locator
// //             .locate_with_mode(&qx, &qy, &mut out, LocateMode::InsideOrBoundary)
// //             .unwrap();
// //         assert!(out[0] >= 0);
// //         assert!(out[1] >= 0);
// //         assert!(out[4] >= 0);
// //     }
// // }
