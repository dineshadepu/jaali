use criterion::{criterion_group, criterion_main, Criterion};
use jaali::{mesh::TriMesh, Bvh2D};

fn generate_grid_mesh_2d(
    nx: usize,
    ny: usize,
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut vx = Vec::with_capacity(nx * ny);
    let mut vy = Vec::with_capacity(nx * ny);

    for j in 0..ny {
        for i in 0..nx {
            vx.push(i as f64);
            vy.push(j as f64);
        }
    }

    let mut t0 = Vec::new();
    let mut t1 = Vec::new();
    let mut t2 = Vec::new();

    let idx = |i: usize, j: usize| j * nx + i;

    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let v00 = idx(i, j);
            let v10 = idx(i + 1, j);
            let v01 = idx(i, j + 1);
            let v11 = idx(i + 1, j + 1);

            // two triangles per cell
            t0.push(v00);
            t1.push(v10);
            t2.push(v11);
            t0.push(v00);
            t1.push(v11);
            t2.push(v01);
        }
    }

    (vx, vy, t0, t1, t2)
}

fn bench_bvh_2d_small_mesh(c: &mut Criterion) {
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

    let queries: Vec<(f64, f64)> = (0..100_000)
        .map(|i| ((i % 100) as f64 * 0.01 + 0.3, 0.3))
        .collect();

    c.bench_function("bvh_2d_small_mesh", |b| {
        b.iter(|| {
            for &(x, y) in &queries {
                std::hint::black_box(bvh.find(x, y, &mesh));
            }
        })
    });
}

fn bench_bvh_2d_large_mesh(c: &mut Criterion) {
    // ~180k triangles
    let (vx, vy, t0, t1, t2) = generate_grid_mesh_2d(300, 300);

    let mesh = TriMesh {
        vx: &vx,
        vy: &vy,
        t0: &t0,
        t1: &t1,
        t2: &t2,
    };
    let bvh = Bvh2D::build(&mesh);

    let queries: Vec<(f64, f64)> = (0..100_000)
        .map(|i| ((i % 300) as f64 + 0.3, ((i / 300) % 300) as f64 + 0.3))
        .collect();

    c.bench_function("bvh_2d_large_mesh", |b| {
        b.iter(|| {
            for &(x, y) in &queries {
                std::hint::black_box(bvh.find(x, y, &mesh));
            }
        })
    });
}

criterion_group!(benches, bench_bvh_2d_small_mesh, bench_bvh_2d_large_mesh);
criterion_main!(benches);
