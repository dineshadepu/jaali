use super::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
fn single_triangle_inside_outside() {
    // Triangle vertices
    let vx = vec![0.0, 1.0, 0.0];
    let vy = vec![0.0, 0.0, 1.0];

    // One triangle
    let t0 = vec![0usize];
    let t1 = vec![1usize];
    let t2 = vec![2usize];

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    let bvh = Bvh2D::build(&mesh);

    // Inside
    let inside = bvh.find(0.25, 0.25, &mesh);
    assert_eq!(inside, 0);

    // Outside
    let outside = bvh.find(1.5, 1.5, &mesh);
    assert_eq!(outside, -1);
}

#[test]
fn single_tet_inside_outside() {
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

    let bvh = Bvh3D::build(&mesh);

    // Inside
    let inside = bvh.find(0.2, 0.2, 0.2, &mesh);
    assert_eq!(inside, 0);

    // Outside
    let outside = bvh.find(1.2, 1.2, 1.2, &mesh);
    assert_eq!(outside, -1);
}

#[test]
fn batch_queries_2d() {
    let vx = vec![0.0, 1.0, 0.0];
    let vy = vec![0.0, 0.0, 1.0];

    let t0 = vec![0usize];
    let t1 = vec![1usize];
    let t2 = vec![2usize];

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    let bvh = Bvh2D::build(&mesh);

    let qx = vec![0.1, 0.8, 2.0];
    let qy = vec![0.1, 0.1, 2.0];

    let mut out = vec![-99; qx.len()];
    for i in 0..qx.len() {
        out[i] = bvh.find(qx[i], qy[i], &mesh);
    }

    assert_eq!(out, vec![0, 0, -1]);
}

pub fn brute_force_find(px: f64, py: f64, mesh: &TriMesh) -> i32 {
    for i in 0..mesh.t0.len() {
        if crate::geometry::point_in_triangle(
            px,
            py,
            mesh.vx[mesh.t0[i]],
            mesh.vy[mesh.t0[i]],
            mesh.vx[mesh.t1[i]],
            mesh.vy[mesh.t1[i]],
            mesh.vx[mesh.t2[i]],
            mesh.vy[mesh.t2[i]],
        ) {
            return i as i32;
        }
    }
    -1
}

#[test]
fn bvh_matches_bruteforce_random_points() {
    // Simple square split into two triangles
    let vx = vec![0.0, 1.0, 1.0, 0.0];
    let vy = vec![0.0, 0.0, 1.0, 1.0];

    let t0 = vec![0usize, 0];
    let t1 = vec![1usize, 2];
    let t2 = vec![2usize, 3];

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    let bvh = Bvh2D::build(&mesh);

    // deterministic pseudo-random points
    let mut x = 0.123_f64;
    let mut y = 0.456_f64;

    for _ in 0..10_000 {
        // simple LCG-style update
        x = (x * 1.37 + 0.11) % 1.5;
        y = (y * 1.91 + 0.07) % 1.5;

        let bvh_id = bvh.find(x, y, &mesh);
        let brute_id = brute_force_find(x, y, &mesh);

        assert_eq!(bvh_id, brute_id);
    }
}

#[test]
fn bvh_boundary_cases() {
    let vx = vec![0.0, 1.0, 0.0];
    let vy = vec![0.0, 0.0, 1.0];

    let t0 = vec![0usize];
    let t1 = vec![1usize];
    let t2 = vec![2usize];

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };
    let bvh = Bvh2D::build(&mesh);

    // vertices → outside in StrictInside
    assert_eq!(bvh.find(0.0, 0.0, &mesh), -1);
    assert_eq!(bvh.find(1.0, 0.0, &mesh), -1);

    // edge midpoint → outside
    assert_eq!(bvh.find(0.5, 0.0, &mesh), -1);

    // strictly inside
    assert_eq!(bvh.find(0.25, 0.25, &mesh), 0);
}

fn vtk_available(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

pub fn read_vtk_2d(filename: &str) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let file = File::open(filename).expect("cannot open file");
    let reader = BufReader::new(file);

    let mut lines = reader.lines().map(|l| l.unwrap());

    // skip header (4 lines)
    for _ in 0..4 {
        lines.next();
    }

    // POINTS line
    let points_line = lines.next().unwrap();
    let parts: Vec<_> = points_line.split_whitespace().collect();
    let n_points: usize = parts[1].parse().unwrap();

    let mut vx = Vec::with_capacity(n_points);
    let mut vy = Vec::with_capacity(n_points);

    // read points
    for _ in 0..n_points {
        let line = lines.next().unwrap();
        let p: Vec<f64> = line
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        vx.push(p[0]);
        vy.push(p[1]); // ignore z
    }

    // CELLS line
    let cells_line = lines.next().unwrap();
    let parts: Vec<_> = cells_line.split_whitespace().collect();
    let n_cells: usize = parts[1].parse().unwrap();

    let mut t0 = Vec::with_capacity(n_cells);
    let mut t1 = Vec::with_capacity(n_cells);
    let mut t2 = Vec::with_capacity(n_cells);

    // read cells
    for _ in 0..n_cells {
        let line = lines.next().unwrap();
        let p: Vec<usize> = line
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();

        debug_assert!(p[0] == 3); // triangle
        t0.push(p[1]);
        t1.push(p[2]);
        t2.push(p[3]);
    }

    // done — ignore rest of file
    (vx, vy, t0, t1, t2)
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let x = ((*seed >> 33) as f64) / ((1u64 << 31) as f64);
    x
}

pub fn generate_points_2d(n: usize, vx: &[f64], vy: &[f64]) -> Vec<(f64, f64)> {
    let (xmin, ymin, xmax, ymax) = mesh_aabb_2d(vx, vy);

    let dx = xmax - xmin;
    let dy = ymax - ymin;

    let sxmin = xmin - 0.5 * dx;
    let symin = ymin - 0.5 * dy;
    let sxmax = xmax + 0.5 * dx;
    let symax = ymax + 0.5 * dy;

    let mut seed = 123456789u64;
    let mut pts = Vec::with_capacity(n);

    for _ in 0..n {
        let x = sxmin + (sxmax - sxmin) * lcg(&mut seed);
        let y = symin + (symax - symin) * lcg(&mut seed);
        pts.push((x, y));
    }

    pts
}

fn mesh_aabb_2d(vx: &[f64], vy: &[f64]) -> (f64, f64, f64, f64) {
    let mut xmin = vx[0];
    let mut ymin = vy[0];
    let mut xmax = vx[0];
    let mut ymax = vy[0];

    for i in 1..vx.len() {
        xmin = xmin.min(vx[i]);
        ymin = ymin.min(vy[i]);
        xmax = xmax.max(vx[i]);
        ymax = ymax.max(vy[i]);
    }
    (xmin, ymin, xmax, ymax)
}

#[test]
#[ignore]
fn stress_bvh_vs_bruteforce_vtk_2d() {
    use std::time::Instant;

    let vtk_path = "./test_data/field_2d.vtk";
    if !vtk_available(vtk_path) {
        eprintln!("VTK file not found, skipping stress test");
        return;
    }

    let (vx, vy, t0, t1, t2) = read_vtk_2d(vtk_path);

    assert!(!vx.is_empty());
    assert_eq!(t0.len(), t1.len());
    assert_eq!(t0.len(), t2.len());

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };

    let bvh = Bvh2D::build(&mesh);

    let n_queries = 1_000;
    let query_points = generate_points_2d(n_queries, &vx, &vy);

    // BVH timing
    let mut bvh_results = Vec::with_capacity(n_queries);
    let t0_bvh = Instant::now();
    for &(px, py) in &query_points {
        bvh_results.push(bvh.find(px, py, &mesh));
    }
    let bvh_time = t0_bvh.elapsed();

    // Brute timing
    let mut brute_results = Vec::with_capacity(n_queries);
    let t0_brute = Instant::now();
    for &(px, py) in &query_points {
        brute_results.push(brute_force_find(px, py, &mesh));
    }
    let brute_time = t0_brute.elapsed();

    // Correctness
    let mismatches = bvh_results
        .iter()
        .zip(&brute_results)
        .filter(|(a, b)| a != b)
        .count();

    assert_eq!(mismatches, 0);

    println!(
        "BVH:   {:.3} s ({:.2} M q/s)",
        bvh_time.as_secs_f64(),
        n_queries as f64 / bvh_time.as_secs_f64() / 1e6
    );
    println!(
        "Brute: {:.3} s ({:.2} M q/s)",
        brute_time.as_secs_f64(),
        n_queries as f64 / brute_time.as_secs_f64() / 1e6
    );
    println!(
        "Speedup: {:.1}×",
        brute_time.as_secs_f64() / bvh_time.as_secs_f64()
    );
}
