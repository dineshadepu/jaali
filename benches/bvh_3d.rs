use criterion::{criterion_group, criterion_main, Criterion};
use jaali::{mesh::TetMesh, Bvh3D};

fn generate_grid_mesh_3d(
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

    let idx = |i: usize, j: usize, k: usize| k * nx * ny + j * nx + i;

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

                t0.push(v000);
                t1.push(v100);
                t2.push(v010);
                t3.push(v001);
            }
        }
    }

    (vx, vy, vz, t0, t1, t2, t3)
}

fn bench_bvh_3d_small_mesh(c: &mut Criterion) {
    // single tetrahedron
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

    let queries: Vec<(f64, f64, f64)> = (0..100_000)
        .map(|i| (0.25, 0.25, (i % 100) as f64 * 0.01))
        .collect();

    c.bench_function("bvh_3d_small_mesh", |b| {
        b.iter(|| {
            for &(x, y, z) in &queries {
                std::hint::black_box(bvh.find(x, y, z, &mesh));
            }
        })
    });
}

fn bench_bvh_3d_large_mesh(c: &mut Criterion) {
    // ~125k tetrahedra
    let (vx, vy, vz, t0, t1, t2, t3) = generate_grid_mesh_3d(50, 50, 50);

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

    let queries: Vec<(f64, f64, f64)> = (0..100_000)
        .map(|i| (25.3, 25.3, (i % 50) as f64 + 0.2))
        .collect();

    c.bench_function("bvh_3d_large_mesh", |b| {
        b.iter(|| {
            for &(x, y, z) in &queries {
                std::hint::black_box(bvh.find(x, y, z, &mesh));
            }
        })
    });
}

criterion_group!(benches, bench_bvh_3d_small_mesh, bench_bvh_3d_large_mesh);
criterion_main!(benches);
