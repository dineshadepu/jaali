// -------------------
// point_in_tet_strict
// -------------------

__device__ __forceinline__
double orient(
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double dx, double dy, double dz
) {
    return (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
         - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
         + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax));
}

__device__ __forceinline__
bool point_in_tet_strict(
    double px, double py, double pz,

    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double dx, double dy, double dz
) {
    const double EPS = 1e-12;

    double v0 = orient(px, py, pz, bx, by, bz, cx, cy, cz, dx, dy, dz);
    double v1 = orient(ax, ay, az, px, py, pz, cx, cy, cz, dx, dy, dz);
    double v2 = orient(ax, ay, az, bx, by, bz, px, py, pz, dx, dy, dz);
    double v3 = orient(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz);
    double v4 = orient(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);

    if (v4 > 0.0) {
        return (v0 > EPS && v1 > EPS && v2 > EPS && v3 > EPS);
    } else {
        return (v0 < -EPS && v1 < -EPS && v2 < -EPS && v3 < -EPS);
    }
}

// ------------------------------------------------------------
// locate_tets kernel
// ------------------------------------------------------------

extern "C" __global__
void locate_tets(
    // Queries
    const double* qx,
    const double* qy,
    const double* qz,
    int* out,
    int n_queries,

    // BVH
    const double* xmin,
    const double* ymin,
    const double* zmin,
    const double* xmax,
    const double* ymax,
    const double* zmax,
    const int* left,
    const int* right,
    const int* tet,

    // Mesh
    const double* vx,
    const double* vy,
    const double* vz,
    const int* t0,
    const int* t1,
    const int* t2,
    const int* t3
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= n_queries) return;

    double px = qx[tidx];
    double py = qy[tidx];
    double pz = qz[tidx];

    // Fixed-size stack (BVH depth bounded)
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int n = stack[--sp];

        // AABB test
        if (px < xmin[n] || px > xmax[n] ||
            py < ymin[n] || py > ymax[n] ||
            pz < zmin[n] || pz > zmax[n]) {
            continue;
        }

        int cell = tet[n];
        if (cell >= 0) {
            int i0 = t0[cell];
            int i1 = t1[cell];
            int i2 = t2[cell];
            int i3 = t3[cell];

            if (point_in_tet_strict(
                px, py, pz,
                vx[i0], vy[i0], vz[i0],
                vx[i1], vy[i1], vz[i1],
                vx[i2], vy[i2], vz[i2],
                vx[i3], vy[i3], vz[i3]
            )) {
                out[tidx] = cell;
                return;
            }
        } else {
            // Internal BVH node
            stack[sp++] = left[n];
            stack[sp++] = right[n];
        }
    }

    out[tidx] = -1;
}
