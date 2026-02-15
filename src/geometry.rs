#[inline(always)]
pub fn point_in_triangle(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> bool {
    let v0x = cx - ax;
    let v0y = cy - ay;
    let v1x = bx - ax;
    let v1y = by - ay;
    let v2x = px - ax;
    let v2y = py - ay;

    let dot00 = v0x * v0x + v0y * v0y;
    let dot01 = v0x * v1x + v0y * v1y;
    let dot02 = v0x * v2x + v0y * v2y;
    let dot11 = v1x * v1x + v1y * v1y;
    let dot12 = v1x * v2x + v1y * v2y;

    let inv = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;

    u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
}

#[inline(always)]
pub fn point_in_triangle_strict(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> bool {
    const EPS: f64 = 1e-12;

    let v0x = cx - ax;
    let v0y = cy - ay;
    let v1x = bx - ax;
    let v1y = by - ay;
    let v2x = px - ax;
    let v2y = py - ay;

    let dot00 = v0x * v0x + v0y * v0y;
    let dot01 = v0x * v1x + v0y * v1y;
    let dot02 = v0x * v2x + v0y * v2y;
    let dot11 = v1x * v1x + v1y * v1y;
    let dot12 = v1x * v2x + v1y * v2y;

    let inv = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;

    u > EPS && v > EPS && (u + v) < 1.0 - EPS
}

#[inline(always)]
pub fn point_in_tet(
    px: f64,
    py: f64,
    pz: f64,
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) -> bool {
    let v = |ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz| {
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
            - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
            + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    };

    let v0 = v(px, py, pz, bx, by, bz, cx, cy, cz, dx, dy, dz);
    let v1 = v(ax, ay, az, px, py, pz, cx, cy, cz, dx, dy, dz);
    let v2 = v(ax, ay, az, bx, by, bz, px, py, pz, dx, dy, dz);
    let v3 = v(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz);
    let v4 = v(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);

    (v0 >= 0.0 && v1 >= 0.0 && v2 >= 0.0 && v3 >= 0.0 && v4 >= 0.0)
        || (v0 <= 0.0 && v1 <= 0.0 && v2 <= 0.0 && v3 <= 0.0 && v4 <= 0.0)
}

#[inline(always)]
pub fn point_in_tet_strict(
    px: f64,
    py: f64,
    pz: f64,

    ax: f64,
    ay: f64,
    az: f64,

    bx: f64,
    by: f64,
    bz: f64,

    cx: f64,
    cy: f64,
    cz: f64,

    dx: f64,
    dy: f64,
    dz: f64,
) -> bool {
    const EPS: f64 = 1e-12;

    #[inline(always)]
    fn orient(
        ax: f64,
        ay: f64,
        az: f64,
        bx: f64,
        by: f64,
        bz: f64,
        cx: f64,
        cy: f64,
        cz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> f64 {
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
            - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
            + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    }

    let v0 = orient(px, py, pz, bx, by, bz, cx, cy, cz, dx, dy, dz);
    let v1 = orient(ax, ay, az, px, py, pz, cx, cy, cz, dx, dy, dz);
    let v2 = orient(ax, ay, az, bx, by, bz, px, py, pz, dx, dy, dz);
    let v3 = orient(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz);
    let v4 = orient(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);

    // All strictly same sign
    if v4 > 0.0 {
        v0 > EPS && v1 > EPS && v2 > EPS && v3 > EPS
    } else {
        v0 < -EPS && v1 < -EPS && v2 < -EPS && v3 < -EPS
    }
}

#[inline(always)]
pub fn point_in_triangle_inclusive(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> bool {
    const EPS: f64 = 1e-12;

    let v0x = cx - ax;
    let v0y = cy - ay;
    let v1x = bx - ax;
    let v1y = by - ay;
    let v2x = px - ax;
    let v2y = py - ay;

    let dot00 = v0x * v0x + v0y * v0y;
    let dot01 = v0x * v1x + v0y * v1y;
    let dot02 = v0x * v2x + v0y * v2y;
    let dot11 = v1x * v1x + v1y * v1y;
    let dot12 = v1x * v2x + v1y * v2y;

    let inv = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;

    u >= -EPS && v >= -EPS && (u + v) <= 1.0 + EPS
}

#[inline(always)]
pub fn point_in_tet_inclusive(
    px: f64,
    py: f64,
    pz: f64,

    ax: f64,
    ay: f64,
    az: f64,

    bx: f64,
    by: f64,
    bz: f64,

    cx: f64,
    cy: f64,
    cz: f64,

    dx: f64,
    dy: f64,
    dz: f64,
) -> bool {
    const EPS: f64 = 1e-12;

    #[inline(always)]
    fn orient(
        ax: f64,
        ay: f64,
        az: f64,
        bx: f64,
        by: f64,
        bz: f64,
        cx: f64,
        cy: f64,
        cz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> f64 {
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
            - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
            + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    }

    let v0 = orient(px, py, pz, bx, by, bz, cx, cy, cz, dx, dy, dz);
    let v1 = orient(ax, ay, az, px, py, pz, cx, cy, cz, dx, dy, dz);
    let v2 = orient(ax, ay, az, bx, by, bz, px, py, pz, dx, dy, dz);
    let v3 = orient(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz);
    let v4 = orient(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);

    if v4 > 0.0 {
        v0 >= -EPS && v1 >= -EPS && v2 >= -EPS && v3 >= -EPS
    } else {
        v0 <= EPS && v1 <= EPS && v2 <= EPS && v3 <= EPS
    }
}
