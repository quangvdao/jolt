pub(super) struct MonomialGeometry {
    pub(super) bits: usize,
    pub(super) weights: usize,
    pub(super) n: usize,
    pub(super) tables: usize,
    pub(super) count: usize,
    pub(super) squared: usize,
}

impl MonomialGeometry {
    pub(super) const fn new(k: usize, nibble_round_2: bool) -> Self {
        assert!(k <= 5);
        let weights = 64 >> (k + 1);
        let width = if k >= 2 || k == 1 && nibble_round_2 {
            4
        } else {
            8
        };
        let bits = if width < weights { width } else { weights };
        let count = 3_usize.pow(k as u32);
        Self {
            bits,
            weights,
            n: 1 << bits,
            tables: weights.div_ceil(bits),
            count,
            squared: count - (1 << k),
        }
    }
}

pub(super) struct WindowGeometry {
    pub(super) width: usize,
    pub(super) n: usize,
    pub(super) a: usize,
    pub(super) c: usize,
    pub(super) units: usize,
}

impl WindowGeometry {
    pub(super) const fn new(k: usize) -> Self {
        assert!(k >= 2 && k <= 5);
        let bound = 1 << k;
        let width = if bound < 8 { bound } else { 8 };
        let a = 64 / width;
        Self {
            width,
            n: 1 << width,
            a,
            c: a / 2,
            units: bound / width,
        }
    }
}
