# Jaali

**Jaali** (Joint Algorithm for Adaptive Localization across Interfaces) is a high-performance, deterministic point-in-cell locator for unstructured meshes, written in Rust. It supports 2D triangular and 3D tetrahedral meshes and runs on serial CPU, multi-core CPU (via Rayon), and GPU (via CUDA) backends — returning identical results across all of them.

---

## Key Features

- **Deterministic across backends** — the same query point returns the same owner cell on serial CPU, parallel CPU, and GPU, even when the point lies on a shared mesh boundary.
- **Multi-hit queries** — `locate_all` returns every cell that geometrically contains the query point; ownership resolution is a separate, order-independent step.
- **Three backends** — `Serial`, `ParallelCPU` (Rayon), and `GPU` (CUDA), selected at runtime.
- **2D and 3D** — triangular meshes (`TriMesh`) and tetrahedral meshes (`TetMesh`).
- **Python bindings** — optional PyO3-based bindings for rapid prototyping.
- **Memory safe** — implemented in Rust with no unsafe code in CPU paths.

---

## Performance

Benchmarked on a mesh of **410,758 tetrahedra**.

### Single-point query

| Method          | Time (ms) | Speedup vs. Brute Force |
| --------------- | --------- | ----------------------- |
| Brute Force     | 1710.48   | 1.00×                   |
| Jaali Serial    | 0.0745    | 22,969×                 |
| Jaali Parallel  | 0.0211    | 80,874×                 |
| Jaali GPU       | 0.1315    | 13,003×                 |

### Batch query (200,000 points)

| Method          | Time (ms) | Speedup vs. Serial |
| --------------- | --------- | ------------------ |
| Jaali Serial    | 673.98    | 1.00×              |
| Jaali Parallel  | 53.03     | 12.71×             |
| Jaali GPU       | 15.26     | 44.18×             |

The single-point speedups reflect BVH acceleration over a brute-force linear scan. The batch speedups show the benefit of parallelism — GPU reaches **44× over serial** at scale, while parallel CPU delivers **12.7×** with no GPU required.

Criterion benchmark suite covers 2D and 3D meshes at small and large scales:

| Benchmark      | Purpose             |
| -------------- | ------------------- |
| 2D small mesh  | per-query overhead  |
| 2D large mesh  | BVH scalability     |
| 3D small mesh  | geometry cost       |
| 3D large mesh  | real-world behavior |

---

## Installation

### Rust

Add Jaali to your `Cargo.toml`:

```toml
[dependencies]
jaali = { path = "path/to/jaali" }
```

Feature flags:

| Flag     | Default | Description                                   |
| -------- | ------- | --------------------------------------------- |
| `rayon`  | yes     | Multi-core CPU backend via Rayon              |
| `python` | no      | Python bindings via PyO3                      |
| `gpu`    | no      | CUDA GPU backend (requires CUDA 13.x toolkit) |

Build with GPU support:

```sh
cargo build --release --features gpu
```

Build without multi-core (serial only):

```sh
cargo build --release --no-default-features
```

### Python

Build and install the Python extension (requires `maturin`):

```sh
pip install maturin
maturin develop --features python          # CPU only
maturin develop --features python,gpu      # with GPU support
```

---

## Rust Usage

### 2D — triangular mesh

```rust
use jaali::mesh::TriMesh;
use jaali::{Backend, Locator2D};

let vx = vec![0.0, 1.0, 1.0, 0.0];
let vy = vec![0.0, 0.0, 1.0, 1.0];
let t0 = vec![0usize, 0];
let t1 = vec![1usize, 2];
let t2 = vec![2usize, 3];

let mesh = TriMesh { vx: &vx, vy: &vy, t0: &t0, t1: &t1, t2: &t2 };

// Build locator on the parallel CPU backend
let mut locator = Locator2D::new(&mesh)
    .with_backend(Backend::ParallelCPU)
    .expect("backend init failed");

let qx = vec![0.3, 0.7];
let qy = vec![0.3, 0.7];

// Single owner per query point (-1 if outside mesh)
let mut out = vec![-1_i32; qx.len()];
locator.locate(&qx, &qy, &mut out);
println!("{:?}", out); // e.g. [0, 1]

// All geometrically valid candidates per query point
locator.locate_all(&qx, &qy).unwrap();
// Results in locator.indices (flattened) and locator.counts
```

### 3D — tetrahedral mesh

```rust
use jaali::mesh::TetMesh;
use jaali::{Backend, Locator3D};

let mesh = TetMesh { vx: &vx, vy: &vy, vz: &vz,
                     t0: &t0, t1: &t1, t2: &t2, t3: &t3 };

let mut locator = Locator3D::new(&mesh)
    .with_backend(Backend::GPU)           // falls back gracefully if GPU unavailable
    .expect("backend init failed");

let mut out = vec![-1_i32; qx.len()];
locator.locate(&qx, &qy, &qz, &mut out);
```

### Switching backends at runtime

```rust
let backend = Backend::Serial;       // or ParallelCPU, GPU
let mut locator = Locator2D::new(&mesh)
    .with_backend(backend)
    .unwrap();
```

---

## Python Usage

```python
import jaali
import numpy as np

# 2D triangular mesh
vx = np.array([0.0, 1.0, 1.0, 0.0])
vy = np.array([0.0, 0.0, 1.0, 1.0])
t0 = np.array([0, 0], dtype=np.intp)
t1 = np.array([1, 2], dtype=np.intp)
t2 = np.array([2, 3], dtype=np.intp)

# backend="serial" | "parallel" (default) | "gpu"
loc = jaali.PyLocator2D(vx.tolist(), vy.tolist(),
                        t0.tolist(), t1.tolist(), t2.tolist(),
                        backend="parallel")

qx = np.array([0.3, 0.7])
qy = np.array([0.3, 0.7])
owner_cells = loc.locate(qx, qy)   # numpy int32 array, -1 = outside mesh
print(owner_cells)

# 3D tetrahedral mesh
loc3d = jaali.PyLocator3D(vx, vy, vz, t0, t1, t2, t3, backend="gpu")
owner_cells = loc3d.locate(qx, qy, qz)

# Query available backends
print(jaali.available_backends())   # ['serial', 'parallel'] or [..., 'gpu']
print(jaali.gpu_available())        # True / False
```

---

## API Reference

### `Backend`

```rust
pub enum Backend { Serial, ParallelCPU, GPU }
```

### `Locator2D` (triangular meshes)

| Method | Description |
|---|---|
| `new(mesh)` | Build BVH; default serial backend |
| `new_with_capacity(mesh, max_queries, max_hits)` | Pre-allocate buffers |
| `with_backend(backend)` | Select execution backend |
| `locate_all(qx, qy)` | Fill `self.indices` and `self.counts` with all candidate cells |
| `locate(qx, qy, out)` | Write single owner cell per query into `out` |

### `Locator3D` (tetrahedral meshes)

Same interface as `Locator2D` with an additional `qz` coordinate parameter.

### Ownership rule

When a query point touches a shared edge, face, or vertex, `locate` returns the candidate cell with the **minimum global cell index**. This rule is order-independent, making results identical across all backends and parallel execution modes.

---

## Determinism

Most point-location implementations rely on ε-tolerances and BVH traversal order to resolve boundary ambiguities. Both vary across CPU and GPU floating-point environments, causing backend-dependent results.

Jaali avoids this by:

1. **Multi-hit traversal** — the BVH collects all geometrically valid candidates without any tolerance cutoff.
2. **Single BVH built on CPU** — transferred unchanged to GPU, so spatial partitioning is identical on all backends.
3. **Order-independent ownership** — minimum cell-index selection is applied as a post-processing step, independent of traversal or scheduling order.

---

## Running Benchmarks

```sh
cargo bench                          # all benchmarks
cargo bench --bench locator_2d       # 2D only
cargo bench --bench locator_3d       # 3D only
cargo bench --features gpu           # include GPU backend
```

Criterion writes HTML reports to `target/criterion/`.

---

## Running Tests

```sh
cargo test                           # all Rust tests
cargo test --features gpu            # include GPU tests

# Python tests (requires maturin develop --features python)
pytest python_tests/
```

Test coverage includes interior points, exterior points, shared edges/faces/vertices, high-valence vertices, large meshes, randomized query distributions, and epsilon sweeps near boundaries — validated against a brute-force reference on all backends.

---

## Applications

- **Lagrangian particle tracking** — map particle positions to mesh cells for field interpolation or coupling force computation.
- **Immersed boundary methods** — identify all cells adjacent to a solid boundary point for distributing coupling terms.
- **FEM/FVM assembly** — locate integration points or boundary nodes in the background mesh.
- **Post-processing** — interpolate fields at arbitrary probe locations in large unstructured meshes.

---

## Citation

If you use Jaali in your research, please cite the associated paper (details to be added on publication).

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
