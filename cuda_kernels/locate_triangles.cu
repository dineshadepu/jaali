extern "C" __device__ __forceinline__
double orient2d(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

extern "C" __device__ __forceinline__
bool point_in_triangle_strict(
    double px, double py,
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {
    const double EPS = 1e-12;

    double v0 = orient2d(px, py, bx, by, cx, cy);
    double v1 = orient2d(ax, ay, px, py, cx, cy);
    double v2 = orient2d(ax, ay, bx, by, px, py);
    double v3 = orient2d(ax, ay, bx, by, cx, cy);

    if (v3 > 0.0) {
        return v0 > EPS && v1 > EPS && v2 > EPS;
    } else {
        return v0 < -EPS && v1 < -EPS && v2 < -EPS;
    }
}

extern "C" __global__
void locate_triangles(
    const double* qx,
    const double* qy,
    int* out,
    int n_queries,

    // BVH
    const double* xmin,
    const double* ymin,
    const double* xmax,
    const double* ymax,
    const int* left,
    const int* right,
    const int* tri,

    // Mesh
    const double* vx,
    const double* vy,
    const int* t0,
    const int* t1,
    const int* t2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_queries) return;

    double px = qx[i];
    double py = qy[i];

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int n = stack[--sp];

        if (px < xmin[n] || px > xmax[n] ||
            py < ymin[n] || py > ymax[n]) {
            continue;
        }

        int tid = tri[n];
        if (tid >= 0) {
            int i0 = t0[tid];
            int i1 = t1[tid];
            int i2 = t2[tid];

            if (point_in_triangle_strict(
                px, py,
                vx[i0], vy[i0],
                vx[i1], vy[i1],
                vx[i2], vy[i2]
            )) {
                out[i] = tid;
                return;
            }
        } else {
            stack[sp++] = left[n];
            stack[sp++] = right[n];
        }
    }

    out[i] = -1;
}
