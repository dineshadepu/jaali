pub struct TriMesh<'a> {
    pub vx: &'a [f64],
    pub vy: &'a [f64],
    pub t0: &'a [usize],
    pub t1: &'a [usize],
    pub t2: &'a [usize],
}

pub struct TetMesh<'a> {
    pub vx: &'a [f64],
    pub vy: &'a [f64],
    pub vz: &'a [f64],
    pub t0: &'a [usize],
    pub t1: &'a [usize],
    pub t2: &'a [usize],
    pub t3: &'a [usize],
}
