import numpy as np
import pytest
import jaali
from utils import single_tet, brute_force_3d


def test_locator3d_basic_inside_all_backends():
    vx, vy, vz, t0, t1, t2, t3 = single_tet()

    qx = np.array([0.1])
    qy = np.array([0.1])
    qz = np.array([0.1])

    for backend in jaali.available_backends():
        locator = jaali.PyLocator3D(
            vx, vy, vz, t0, t1, t2, t3,
            backend=backend
        )

        out = locator.locate(qx, qy, qz)
        assert out[0] == 0, f"Failed for backend {backend}"


@pytest.mark.slow
def test_stress_locator3d_vs_bruteforce():
    # Regular grid tetrahedral mesh
    nx, ny, nz = 25, 25, 25

    xs, ys, zs = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij"
    )

    vx = xs.flatten().astype(float)
    vy = ys.flatten().astype(float)
    vz = zs.flatten().astype(float)

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    t0, t1, t2, t3 = [], [], [], []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                v000 = idx(i, j, k)
                v100 = idx(i + 1, j, k)
                v010 = idx(i, j + 1, k)
                v001 = idx(i, j, k + 1)
                v111 = idx(i + 1, j + 1, k + 1)

                # Simple tet decomposition (not optimal, but deterministic)
                t0.append(v000); t1.append(v100); t2.append(v010); t3.append(v001)
                t0.append(v111); t1.append(v100); t2.append(v010); t3.append(v001)

    t0 = np.array(t0, dtype=np.int64)
    t1 = np.array(t1, dtype=np.int64)
    t2 = np.array(t2, dtype=np.int64)
    t3 = np.array(t3, dtype=np.int64)

    # Queries strictly inside
    qx = np.random.uniform(1.0, nx - 2.0, 30)
    qy = np.random.uniform(1.0, ny - 2.0, 30)
    qz = np.random.uniform(1.0, nz - 2.0, 30)

    brute = np.array([
        brute_force_3d(x, y, z, vx, vy, vz, t0, t1, t2, t3)
        for x, y, z in zip(qx, qy, qz)
    ])

    for backend in jaali.available_backends():
        locator = jaali.PyLocator3D(
            vx, vy, vz, t0, t1, t2, t3,
            backend=backend
        )

        out = locator.locate(qx, qy, qz)

        mismatches = np.sum((out >= 0) != (brute >= 0))
        assert mismatches == 0, f"Mismatches for backend {backend}"

    # ==============================
    # Check among different backends
    # ==============================
    # Queries strictly inside
    qx = np.random.uniform(1.0, nx - 2.0, 3000)
    qy = np.random.uniform(1.0, ny - 2.0, 3000)
    qz = np.random.uniform(1.0, nz - 2.0, 3000)

    # Always compare CPU backends
    locator_serial = jaali.PyLocator3D(
        vx, vy, vz, t0, t1, t2, t3, backend="serial"
    )
    locator_parallel = jaali.PyLocator3D(
        vx, vy, vz, t0, t1, t2, t3, backend="parallel"
    )

    out_serial = locator_serial.locate(qx, qy, qz)
    out_parallel = locator_parallel.locate(qx, qy, qz)

    assert np.all((out_serial >= 0) == (out_parallel >= 0))

    # Only compare GPU if available
    if "gpu" in jaali.available_backends():
        locator_gpu = jaali.PyLocator3D(
            vx, vy, vz, t0, t1, t2, t3, backend="gpu"
        )
        out_gpu = locator_gpu.locate(qx, qy, qz)

        assert np.all((out_serial >= 0) == (out_gpu >= 0))
