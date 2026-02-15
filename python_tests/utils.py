import numpy as np
import jaali


def single_triangle():
    vx = np.array([0.0, 1.0, 0.0])
    vy = np.array([0.0, 0.0, 1.0])
    t0 = np.array([0], dtype=np.int64)
    t1 = np.array([1], dtype=np.int64)
    t2 = np.array([2], dtype=np.int64)
    return vx, vy, t0, t1, t2


def single_tet():
    vx = np.array([0.0, 1.0, 0.0, 0.0])
    vy = np.array([0.0, 0.0, 1.0, 0.0])
    vz = np.array([0.0, 0.0, 0.0, 1.0])
    t0 = np.array([0], dtype=np.int64)
    t1 = np.array([1], dtype=np.int64)
    t2 = np.array([2], dtype=np.int64)
    t3 = np.array([3], dtype=np.int64)
    return vx, vy, vz, t0, t1, t2, t3


def orient2d(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def point_in_triangle_strict(px, py, ax, ay, bx, by, cx, cy):
    eps = 1e-12

    v0 = orient2d(px, py, bx, by, cx, cy)
    v1 = orient2d(ax, ay, px, py, cx, cy)
    v2 = orient2d(ax, ay, bx, by, px, py)
    v3 = orient2d(ax, ay, bx, by, cx, cy)

    if v3 > 0:
        return v0 > eps and v1 > eps and v2 > eps
    else:
        return v0 < -eps and v1 < -eps and v2 < -eps


def brute_force_2d(px, py, vx, vy, t0, t1, t2):
    for i in range(len(t0)):
        if point_in_triangle_strict(
            px,
            py,
            vx[t0[i]],
            vy[t0[i]],
            vx[t1[i]],
            vy[t1[i]],
            vx[t2[i]],
            vy[t2[i]],
        ):
            return i
    return -1


# ---------------------------------------
# Brute force point-in-tet (STRICT)
# ---------------------------------------
def orient3d(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz):
    return (
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
        - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
        + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    )


def point_in_tet_strict(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz):
    eps = 1e-12

    v0 = orient3d(px, py, pz, bx, by, bz, cx, cy, cz, dx, dy, dz)
    v1 = orient3d(ax, ay, az, px, py, pz, cx, cy, cz, dx, dy, dz)
    v2 = orient3d(ax, ay, az, bx, by, bz, px, py, pz, dx, dy, dz)
    v3 = orient3d(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz)
    v4 = orient3d(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz)

    if v4 > 0:
        return v0 > eps and v1 > eps and v2 > eps and v3 > eps
    else:
        return v0 < -eps and v1 < -eps and v2 < -eps and v3 < -eps


def brute_force_3d(px, py, pz, vx, vy, vz, t0, t1, t2, t3):
    for i in range(len(t0)):
        if point_in_tet_strict(
            px,
            py,
            pz,
            vx[t0[i]],
            vy[t0[i]],
            vz[t0[i]],
            vx[t1[i]],
            vy[t1[i]],
            vz[t1[i]],
            vx[t2[i]],
            vy[t2[i]],
            vz[t2[i]],
            vx[t3[i]],
            vy[t3[i]],
            vz[t3[i]],
        ):
            return i
    return -1
