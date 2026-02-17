use criterion::{criterion_group, criterion_main, Criterion};
use jaali::mesh::TetMesh;
use jaali::{Backend, Locator3D};

fn available_backends() -> Vec<Backend> {
    let mut v = vec![Backend::Serial, Backend::ParallelCPU];

    #[cfg(feature = "gpu")]
    {
        v.push(Backend::GPU);
    }

    v
}

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

    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                vx.push(i as f64);
                vy.push(j as f64);
                vz.push(k as f64);
            }
        }
    }

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
                let v111 = idx(i + 1, j + 1, k + 1);

                // two tets per cube (simple, consistent)
                t0.push(v000);
                t1.push(v100);
                t2.push(v010);
                t3.push(v111);

                t0.push(v000);
                t1.push(v010);
                t2.push(v001);
                t3.push(v111);
            }
        }
    }

    (vx, vy, vz, t0, t1, t2, t3)
}

fn bench_locator_3d_small_mesh(c: &mut Criterion) {
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

    let qx: Vec<f64> = vec![0.1; 100_000];
    let qy: Vec<f64> = vec![0.1; 100_000];
    let qz: Vec<f64> = vec![0.1; 100_000];

    for backend in available_backends() {
        // ---------- locate_all ----------
        let mut locator = Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        let name = format!("locator3d_small_locate_all_{:?}", backend);

        c.bench_function(&name, |b| {
            b.iter(|| {
                locator.locate_all(&qx, &qy, &qz).unwrap();
                std::hint::black_box(&locator.indices);
            })
        });

        // ---------- locate ----------
        let mut locator = Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        let mut out = vec![-1; qx.len()];
        let name = format!("locator3d_small_locate_{:?}", backend);

        c.bench_function(&name, |b| {
            b.iter(|| {
                locator.locate(&qx, &qy, &qz, &mut out);
                std::hint::black_box(&out);
            })
        });
    }
}

fn bench_locator_3d_large_mesh(c: &mut Criterion) {
    let (vx, vy, vz, t0, t1, t2, t3) = generate_grid_mesh_3d(40, 40, 40);

    let mesh = TetMesh {
        vx: &vx,
        vy: &vy,
        vz: &vz,
        t0: &t0,
        t1: &t1,
        t2: &t2,
        t3: &t3,
    };

    let n = 100_000;
    let qx: Vec<f64> = (0..n).map(|i| (i % 40) as f64 + 0.3).collect();
    let qy: Vec<f64> = (0..n).map(|i| ((i / 40) % 40) as f64 + 0.3).collect();
    let qz: Vec<f64> = (0..n).map(|i| ((i / 1600) % 40) as f64 + 0.3).collect();

    for backend in available_backends() {
        // ---------- locate_all ----------
        let mut locator = Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        let name = format!("locator3d_large_locate_all_{:?}", backend);

        c.bench_function(&name, |b| {
            b.iter(|| {
                locator.locate_all(&qx, &qy, &qz).unwrap();
                std::hint::black_box(&locator.indices);
            })
        });

        // ---------- locate ----------
        let mut locator = Locator3D::new(&mesh)
            .with_backend(backend)
            .expect("backend init failed");

        let mut out = vec![-1; qx.len()];
        let name = format!("locator3d_large_locate_{:?}", backend);

        c.bench_function(&name, |b| {
            b.iter(|| {
                locator.locate(&qx, &qy, &qz, &mut out);
                std::hint::black_box(&out);
            })
        });
    }
}

criterion_group!(
    benches,
    bench_locator_3d_small_mesh,
    bench_locator_3d_large_mesh
);
criterion_main!(benches);
