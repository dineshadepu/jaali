from utils import single_triangle, brute_force_2d
import pytest
import jaali
import numpy as np


def test_locator2d_basic_inside_all_backends():
    # Single triangle
    vx, vy, t0, t1, t2 = single_triangle()
    qx = np.array([0.25])
    qy = np.array([0.25])

    for backend in jaali.available_backends():
        locator = jaali.PyLocator2D(vx, vy, t0, t1, t2, backend=backend)
        out = locator.locate(qx, qy)

        assert out[0] == 0, f"Failed for backend {backend}"


@pytest.mark.slow
def test_stress_locator2d_vs_bruteforce():
    nx, ny = 150, 150
    vx, vy = np.meshgrid(np.arange(nx), np.arange(ny))
    vx = vx.flatten().astype(float)
    vy = vy.flatten().astype(float)

    t0, t1, t2 = [], [], []
    def idx(i, j): return j * nx + i

    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v01 = idx(i, j + 1)
            v11 = idx(i + 1, j + 1)

            t0 += [v00, v00]
            t1 += [v10, v11]
            t2 += [v11, v01]

    t0 = np.array(t0, dtype=np.int64)
    t1 = np.array(t1, dtype=np.int64)
    t2 = np.array(t2, dtype=np.int64)

    qx = np.random.uniform(1, nx - 2, 30)
    qy = np.random.uniform(1, ny - 2, 30)

    brute = np.array([
        brute_force_2d(x, y, vx, vy, t0, t1, t2)
        for x, y in zip(qx, qy)
    ])

    for backend in jaali.available_backends():
        locator = jaali.PyLocator2D(vx, vy, t0, t1, t2, backend=backend)
        out = locator.locate(qx, qy)

        mismatches = np.sum((out >= 0) != (brute >= 0))
        assert mismatches == 0, f"Mismatches for backend {backend}"

    # ==============================
    # Check among different backends
    # ==============================
    # Queries strictly inside
    qx = np.random.uniform(1.0, nx - 2.0, 3000)
    qy = np.random.uniform(1.0, ny - 2.0, 3000)

    # Always compare CPU backends
    locator_serial = jaali.PyLocator2D(
        vx, vy, t0, t1, t2, backend="serial"
    )
    locator_parallel = jaali.PyLocator2D(
        vx, vy, t0, t1, t2, backend="parallel"
    )

    out_serial = locator_serial.locate(qx, qy)
    out_parallel = locator_parallel.locate(qx, qy)

    assert np.all((out_serial >= 0) == (out_parallel >= 0))

    # Only compare GPU if available
    if "gpu" in jaali.available_backends():
        locator_gpu = jaali.PyLocator2D(
            vx, vy, t0, t1, t2, backend="gpu"
        )
        out_gpu = locator_gpu.locate(qx, qy)

        assert np.all((out_serial >= 0) == (out_gpu >= 0))
