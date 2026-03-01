/// PCS-agnostic matrix geometry for laying out 1D polynomial coefficients
/// into a 2D matrix (rows × columns).
///
/// For Dory, this is derived from `DoryGlobals`. For Hachi, it will be
/// derived from ring packing parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub num_columns: usize,
    pub num_rows: usize,
    /// Total number of cycles (T) in the OneHot polynomial.
    pub T: usize,
    /// Whether the layout is cycle-major (true) or address-major (false).
    pub cycle_major: bool,
}

impl MatrixLayout {
    /// Convert a (address, cycle) pair to a flat coefficient index.
    pub fn address_cycle_to_index(
        &self,
        address: usize,
        cycle: usize,
        K: usize,
        T: usize,
    ) -> usize {
        if self.cycle_major {
            address * T + cycle
        } else {
            cycle * K + address
        }
    }

    /// Number of cycles per row in address-major layout.
    /// In cycle-major layout this is not meaningful and panics.
    pub fn cycles_per_row(&self) -> usize {
        assert!(
            !self.cycle_major,
            "cycles_per_row is only defined for address-major layout"
        );
        let k = self.num_rows * self.num_columns / self.T;
        debug_assert!(k > 0);
        self.num_columns / k
    }

    /// Compute the number of rows needed for a OneHot polynomial with
    /// `K` addresses and `T` cycles.
    pub fn one_hot_num_rows(&self, K: usize, T: usize) -> usize {
        if self.cycle_major {
            (T * K).div_ceil(self.num_columns)
        } else {
            T.div_ceil(self.cycles_per_row())
        }
    }
}
