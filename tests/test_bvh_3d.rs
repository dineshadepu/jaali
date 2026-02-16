// use jaali::*;
// use std::fs::File;
// use std::io::{BufRead, BufReader};

// fn read_vtk_3d(
//     filename: &str,
// ) -> (
//     Vec<f64>,
//     Vec<f64>,
//     Vec<f64>,
//     Vec<usize>,
//     Vec<usize>,
//     Vec<usize>,
//     Vec<usize>,
// ) {
//     let file = File::open(filename).expect("cannot open VTK file");
//     let reader = BufReader::new(file);
//     let mut lines = reader.lines().map(|l| l.unwrap());

//     // Skip header (4 lines)
//     for _ in 0..4 {
//         lines.next();
//     }

//     // POINTS
//     let line = lines.next().unwrap();
//     let parts: Vec<&str> = line.split_whitespace().collect();
//     let n_points: usize = parts[1].parse().unwrap();

//     let mut vx = Vec::with_capacity(n_points);
//     let mut vy = Vec::with_capacity(n_points);
//     let mut vz = Vec::with_capacity(n_points);

//     for _ in 0..n_points {
//         let p: Vec<f64> = lines
//             .next()
//             .unwrap()
//             .split_whitespace()
//             .map(|x| x.parse().unwrap())
//             .collect();
//         vx.push(p[0]);
//         vy.push(p[1]);
//         vz.push(p[2]);
//     }

//     // CELLS
//     let line = lines.next().unwrap();
//     let parts: Vec<&str> = line.split_whitespace().collect();
//     let n_cells: usize = parts[1].parse().unwrap();

//     let mut t0 = Vec::with_capacity(n_cells);
//     let mut t1 = Vec::with_capacity(n_cells);
//     let mut t2 = Vec::with_capacity(n_cells);
//     let mut t3 = Vec::with_capacity(n_cells);

//     for _ in 0..n_cells {
//         let c: Vec<usize> = lines
//             .next()
//             .unwrap()
//             .split_whitespace()
//             .map(|x| x.parse().unwrap())
//             .collect();
//         debug_assert!(c[0] == 4); // tetra
//         t0.push(c[1]);
//         t1.push(c[2]);
//         t2.push(c[3]);
//         t3.push(c[4]);
//     }

//     (vx, vy, vz, t0, t1, t2, t3)
// }

// fn mesh_aabb_3d(vx: &[f64], vy: &[f64], vz: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
//     let mut xmin = vx[0];
//     let mut ymin = vy[0];
//     let mut zmin = vz[0];
//     let mut xmax = xmin;
//     let mut ymax = ymin;
//     let mut zmax = zmin;

//     for i in 1..vx.len() {
//         xmin = xmin.min(vx[i]);
//         ymin = ymin.min(vy[i]);
//         zmin = zmin.min(vz[i]);
//         xmax = xmax.max(vx[i]);
//         ymax = ymax.max(vy[i]);
//         zmax = zmax.max(vz[i]);
//     }
//     (xmin, ymin, zmin, xmax, ymax, zmax)
// }

// fn generate_points_3d(n: usize, vx: &[f64], vy: &[f64], vz: &[f64]) -> Vec<(f64, f64, f64)> {
//     let (xmin, ymin, zmin, xmax, ymax, zmax) = mesh_aabb_3d(vx, vy, vz);

//     let dx = xmax - xmin;
//     let dy = ymax - ymin;
//     let dz = zmax - zmin;

//     let sxmin = xmin - 0.5 * dx;
//     let symin = ymin - 0.5 * dy;
//     let szmin = zmin - 0.5 * dz;
//     let sxmax = xmax + 0.5 * dx;
//     let symax = ymax + 0.5 * dy;
//     let szmax = zmax + 0.5 * dz;

//     let mut seed = 123456789u64;
//     let mut pts = Vec::with_capacity(n);

//     for _ in 0..n {
//         let x = sxmin + (sxmax - sxmin) * lcg(&mut seed);
//         let y = symin + (symax - symin) * lcg(&mut seed);
//         let z = szmin + (szmax - szmin) * lcg(&mut seed);
//         pts.push((x, y, z));
//     }
//     pts
// }

// fn lcg(seed: &mut u64) -> f64 {
//     *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
//     ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
// }

// fn brute_force_find_tet(px: f64, py: f64, pz: f64, mesh: &TetMesh) -> i32 {
//     for i in 0..mesh.t0.len() {
//         if crate::geometry::point_in_tet(
//             px,
//             py,
//             pz,
//             mesh.vx[mesh.t0[i]],
//             mesh.vy[mesh.t0[i]],
//             mesh.vz[mesh.t0[i]],
//             mesh.vx[mesh.t1[i]],
//             mesh.vy[mesh.t1[i]],
//             mesh.vz[mesh.t1[i]],
//             mesh.vx[mesh.t2[i]],
//             mesh.vy[mesh.t2[i]],
//             mesh.vz[mesh.t2[i]],
//             mesh.vx[mesh.t3[i]],
//             mesh.vy[mesh.t3[i]],
//             mesh.vz[mesh.t3[i]],
//         ) {
//             return i as i32;
//         }
//     }
//     -1
// }

// #[test]
// fn single_tet_inside_outside() {
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
//     let bvh = Bvh3D::build(&mesh);

//     assert_eq!(bvh.find_strict(0.2, 0.2, 0.2, &mesh), 0);
//     assert_eq!(bvh.find_strict(1.2, 1.2, 1.2, &mesh), -1);
// }

// #[test]
// fn tet_boundary_cases() {
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
//     let bvh = Bvh3D::build(&mesh);

//     // Vertex
//     assert_eq!(bvh.find_strict(0.0, 0.0, 0.0, &mesh), -1);

//     // Face center
//     assert_eq!(bvh.find_strict(0.33, 0.33, 0.0, &mesh), -1);

//     // Edge midpoint
//     assert_eq!(bvh.find_strict(0.5, 0.0, 0.0, &mesh), -1);

//     // strictly inside
//     assert_eq!(bvh.find_strict(0.2, 0.2, 0.2, &mesh), 0);
// }

// #[test]
// fn bvh_3d_matches_bruteforce_random() {
//     // Two tets forming a cube corner
//     let vx = vec![0.0, 1.0, 0.0, 0.0, 1.0];
//     let vy = vec![0.0, 0.0, 1.0, 0.0, 1.0];
//     let vz = vec![0.0, 0.0, 0.0, 1.0, 1.0];

//     let t0 = vec![0usize, 1];
//     let t1 = vec![1usize, 2];
//     let t2 = vec![2usize, 3];
//     let t3 = vec![3usize, 4];

//     let mesh = TetMesh {
//         vx: &vx,
//         vy: &vy,
//         vz: &vz,
//         t0: &t0,
//         t1: &t1,
//         t2: &t2,
//         t3: &t3,
//     };
//     let bvh = Bvh3D::build(&mesh);

//     let mut x = 0.17;
//     let mut y = 0.41;
//     let mut z = 0.73;

//     for _ in 0..20_000 {
//         x = (x * 1.31 + 0.11) % 1.5;
//         y = (y * 1.73 + 0.07) % 1.5;
//         z = (z * 1.19 + 0.13) % 1.5;

//         let bvh_id = bvh.find_strict(x, y, z, &mesh);
//         let brute_id = brute_force_find_tet(x, y, z, &mesh);

//         assert_eq!(bvh_id, brute_id);
//     }
// }

// // #[test]
// // #[ignore]
// // fn stress_bvh_vs_bruteforce_vtk_3d() {
// //     let vtk_path = "./test_data/field_3d.vtk";
// //     if !std::path::Path::new(vtk_path).exists() {
// //         eprintln!("VTK file not found, skipping 3D stress test");
// //         return;
// //     }

// //     let (vx, vy, vz, t0, t1, t2, t3) = read_vtk_3d(vtk_path);

// //     let mesh = TetMesh {
// //         vx: &vx,
// //         vy: &vy,
// //         vz: &vz,
// //         t0: &t0,
// //         t1: &t1,
// //         t2: &t2,
// //         t3: &t3,
// //     };
// //     let bvh = Bvh3D::build(&mesh);

// //     let n_queries = 50;
// //     let queries = generate_points_3d(n_queries, &vx, &vy, &vz);

// //     let mut bvh_out = Vec::with_capacity(n_queries);
// //     let mut brute_out = Vec::with_capacity(n_queries);

// //     use std::time::Instant;

// //     let t0 = Instant::now();
// //     for &(x, y, z) in &queries {
// //         bvh_out.push(bvh.find_strict(x, y, z, &mesh));
// //     }
// //     let bvh_time = t0.elapsed();

// //     let t1 = Instant::now();
// //     for &(x, y, z) in &queries {
// //         brute_out.push(brute_force_find_tet(x, y, z, &mesh));
// //     }
// //     let brute_time = t1.elapsed();

// //     assert_eq!(
// //         bvh_out
// //             .iter()
// //             .zip(&brute_out)
// //             .filter(|(a, b)| a != b)
// //             .count(),
// //         0
// //     );

// //     println!(
// //         "3D BVH speedup: {:.1}×",
// //         brute_time.as_secs_f64() / bvh_time.as_secs_f64()
// //     );
// // }
