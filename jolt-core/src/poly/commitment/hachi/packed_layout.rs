use hachi_pcs::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, optimal_m_r_split, CommitmentConfig,
    HachiCommitmentLayout,
};

/// Avoid degenerate layouts that tile only across polynomials and barely any
/// cycles; that would defeat the trace-locality goal of the packed layout.
const PACKED_MIN_CYCLE_TILE_BITS: usize = 6;

/// Cap the number of inner poly bits so each block spans a modest number of
/// polynomials instead of turning every block into an all-polys mega-tile.
const PACKED_MAX_POLY_TILE_BITS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedPosition {
    pub block_idx: usize,
    pub pos_in_block: usize,
    pub coeff_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedBlockRange {
    pub cycle_start: usize,
    pub cycle_end: usize,
    pub poly_start: usize,
    pub poly_end: usize,
}

/// Bit layout for the packed one-hot polynomial after the first `alpha = log2(D)`
/// coefficient bits have been selected.
///
/// For `K >= D`, all `alpha` coefficient bits come from the one-hot address and
/// `addr_inner_bits()` carries any remaining address bits into the reduced ring
/// index. For `K < D`, only `log_k` coefficient bits come from the address and
/// the remaining `alpha - log_k` bits are lifted from low cycle/poly bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedBitLayout {
    alpha_bits: usize,
    log_k: usize,
    log_t: usize,
    log_packed: usize,
    cycle_coeff_bits: usize,
    poly_coeff_bits: usize,
    cycle_inner_bits: usize,
    poly_inner_bits: usize,
}

impl PackedBitLayout {
    pub(super) fn new<const D: usize>(
        log_k: usize,
        log_t: usize,
        log_packed: usize,
        cycle_coeff_bits: usize,
        poly_coeff_bits: usize,
        cycle_inner_bits: usize,
        poly_inner_bits: usize,
    ) -> Self {
        let alpha_bits = D.trailing_zeros() as usize;
        let addr_coeff_bits = log_k.min(alpha_bits);
        assert!(
            cycle_coeff_bits + poly_coeff_bits == alpha_bits - addr_coeff_bits,
            "packed layout coeff split mismatch (cycle_coeff_bits={cycle_coeff_bits}, poly_coeff_bits={poly_coeff_bits}, alpha={alpha_bits}, log_k={log_k})"
        );
        assert!(
            cycle_coeff_bits + cycle_inner_bits <= log_t,
            "cycle coeff/inner bits exceed log_t (cycle_coeff_bits={cycle_coeff_bits}, cycle_inner_bits={cycle_inner_bits}, log_t={log_t})"
        );
        assert!(
            poly_coeff_bits + poly_inner_bits <= log_packed,
            "poly coeff/inner bits exceed log_packed (poly_coeff_bits={poly_coeff_bits}, poly_inner_bits={poly_inner_bits}, log_packed={log_packed})"
        );
        Self {
            alpha_bits,
            log_k,
            log_t,
            log_packed,
            cycle_coeff_bits,
            poly_coeff_bits,
            cycle_inner_bits,
            poly_inner_bits,
        }
    }

    #[inline]
    pub(super) fn log_k(&self) -> usize {
        self.log_k
    }

    #[cfg(test)]
    pub(super) fn cycle_coeff_bits(&self) -> usize {
        self.cycle_coeff_bits
    }

    #[cfg(test)]
    pub(super) fn poly_coeff_bits(&self) -> usize {
        self.poly_coeff_bits
    }

    #[cfg(test)]
    pub(super) fn cycle_inner_bits(&self) -> usize {
        self.cycle_inner_bits
    }

    #[cfg(test)]
    pub(super) fn poly_inner_bits(&self) -> usize {
        self.poly_inner_bits
    }

    #[inline]
    pub(super) fn addr_coeff_bits(&self) -> usize {
        self.log_k.min(self.alpha_bits)
    }

    #[inline]
    pub(super) fn addr_inner_bits(&self) -> usize {
        self.log_k - self.addr_coeff_bits()
    }

    #[inline]
    pub(super) fn lifted_coeff_bits(&self) -> usize {
        self.cycle_coeff_bits + self.poly_coeff_bits
    }

    #[inline]
    pub(super) fn cycle_tile_bits(&self) -> usize {
        self.cycle_coeff_bits + self.cycle_inner_bits
    }

    #[inline]
    pub(super) fn poly_tile_bits(&self) -> usize {
        self.poly_coeff_bits + self.poly_inner_bits
    }

    #[inline]
    pub(super) fn cycle_outer_bits(&self) -> usize {
        self.log_t - self.cycle_tile_bits()
    }

    #[inline]
    pub(super) fn poly_outer_bits(&self) -> usize {
        self.log_packed - self.poly_tile_bits()
    }

    #[inline]
    pub(super) fn m_vars(&self) -> usize {
        self.addr_inner_bits() + self.cycle_inner_bits + self.poly_inner_bits
    }

    #[inline]
    pub(super) fn r_vars(&self) -> usize {
        self.cycle_outer_bits() + self.poly_outer_bits()
    }

    #[inline]
    pub(super) fn block_len(&self) -> usize {
        1usize
            .checked_shl(self.m_vars() as u32)
            .expect("packed block_len overflow")
    }

    #[inline]
    pub(super) fn num_blocks(&self) -> usize {
        1usize
            .checked_shl(self.r_vars() as u32)
            .expect("packed num_blocks overflow")
    }

    #[inline]
    pub(super) fn num_padded_polys(&self) -> usize {
        1usize
            .checked_shl(self.log_packed as u32)
            .expect("packed num_padded overflow")
    }

    #[inline]
    pub(super) fn total_num_vars(&self) -> usize {
        self.alpha_bits + self.m_vars() + self.r_vars()
    }

    #[inline]
    pub(super) fn max_coeffs_per_entry(&self) -> usize {
        1usize
            .checked_shl(self.lifted_coeff_bits() as u32)
            .expect("packed max_coeffs_per_entry overflow")
    }

    #[inline]
    pub(super) fn into_hachi_layout<Cfg: CommitmentConfig>(
        self,
        log_basis: u32,
    ) -> HachiCommitmentLayout {
        HachiCommitmentLayout::new_with_decomp(
            self.m_vars(),
            self.r_vars(),
            Cfg::N_A,
            1,
            compute_num_digits(128, log_basis),
            compute_num_digits_fold(self.r_vars(), Cfg::CHALLENGE_WEIGHT, log_basis),
            log_basis,
        )
        .expect("invalid packed Hachi layout")
    }

    #[inline]
    pub(super) fn locate(
        &self,
        cycle_idx: usize,
        poly_idx: usize,
        hot_index: usize,
    ) -> PackedPosition {
        let addr_coeff_bits = self.addr_coeff_bits();
        let addr_coeff_mask = if addr_coeff_bits == 0 {
            0
        } else {
            (1usize << addr_coeff_bits) - 1
        };
        let addr_coeff = hot_index & addr_coeff_mask;
        let addr_inner = hot_index >> addr_coeff_bits;
        debug_assert!(addr_inner < (1usize << self.addr_inner_bits()));

        let cycle_coeff_mask = if self.cycle_coeff_bits == 0 {
            0
        } else {
            (1usize << self.cycle_coeff_bits) - 1
        };
        let cycle_coeff = cycle_idx & cycle_coeff_mask;
        let cycle_shifted = cycle_idx >> self.cycle_coeff_bits;
        let cycle_inner_mask = if self.cycle_inner_bits == 0 {
            0
        } else {
            (1usize << self.cycle_inner_bits) - 1
        };
        let cycle_inner = cycle_shifted & cycle_inner_mask;
        let cycle_outer = cycle_shifted >> self.cycle_inner_bits;

        let poly_coeff_mask = if self.poly_coeff_bits == 0 {
            0
        } else {
            (1usize << self.poly_coeff_bits) - 1
        };
        let poly_coeff = poly_idx & poly_coeff_mask;
        let poly_shifted = poly_idx >> self.poly_coeff_bits;
        let poly_inner_mask = if self.poly_inner_bits == 0 {
            0
        } else {
            (1usize << self.poly_inner_bits) - 1
        };
        let poly_inner = poly_shifted & poly_inner_mask;
        let poly_outer = poly_shifted >> self.poly_inner_bits;

        let coeff_idx = addr_coeff
            | (cycle_coeff << addr_coeff_bits)
            | (poly_coeff << (addr_coeff_bits + self.cycle_coeff_bits));
        let pos_in_block = addr_inner
            | (cycle_inner << self.addr_inner_bits())
            | (poly_inner << (self.addr_inner_bits() + self.cycle_inner_bits));
        let block_idx = cycle_outer | (poly_outer << self.cycle_outer_bits());

        PackedPosition {
            block_idx,
            pos_in_block,
            coeff_idx,
        }
    }

    #[inline]
    pub(super) fn block_range(
        &self,
        block_idx: usize,
        num_cycles: usize,
        num_polys: usize,
    ) -> PackedBlockRange {
        let cycle_outer_bits = self.cycle_outer_bits();
        let cycle_outer_mask = if cycle_outer_bits == 0 {
            0
        } else {
            (1usize << cycle_outer_bits) - 1
        };
        let cycle_outer = if cycle_outer_bits == 0 {
            0
        } else {
            block_idx & cycle_outer_mask
        };
        let poly_outer = if cycle_outer_bits == 0 {
            block_idx
        } else {
            block_idx >> cycle_outer_bits
        };

        let cycle_start = cycle_outer << self.cycle_tile_bits();
        let cycle_end = (cycle_start + (1usize << self.cycle_tile_bits())).min(num_cycles);
        let poly_start = poly_outer << self.poly_tile_bits();
        let poly_end = (poly_start + (1usize << self.poly_tile_bits())).min(num_polys);

        PackedBlockRange {
            cycle_start,
            cycle_end,
            poly_start,
            poly_end,
        }
    }

    pub(super) fn reorder_packed_point<F: Copy>(
        &self,
        cycle_bits_le: &[F],
        addr_bits_le: &[F],
        poly_bits_le: &[F],
    ) -> Vec<F> {
        assert_eq!(
            cycle_bits_le.len(),
            self.log_t,
            "cycle_bits length mismatch (expected {}, got {got})",
            self.log_t,
            got = cycle_bits_le.len()
        );
        assert_eq!(
            addr_bits_le.len(),
            self.log_k,
            "addr_bits length mismatch (expected {}, got {got})",
            self.log_k,
            got = addr_bits_le.len()
        );
        assert_eq!(
            poly_bits_le.len(),
            self.log_packed,
            "poly_bits length mismatch (expected {}, got {got})",
            self.log_packed,
            got = poly_bits_le.len()
        );

        let mut out = Vec::with_capacity(self.total_num_vars());
        let addr_coeff_bits = self.addr_coeff_bits();
        out.extend_from_slice(&addr_bits_le[..addr_coeff_bits]);
        out.extend_from_slice(&cycle_bits_le[..self.cycle_coeff_bits]);
        out.extend_from_slice(&poly_bits_le[..self.poly_coeff_bits]);
        out.extend_from_slice(&addr_bits_le[addr_coeff_bits..]);
        out.extend_from_slice(
            &cycle_bits_le[self.cycle_coeff_bits..self.cycle_coeff_bits + self.cycle_inner_bits],
        );
        out.extend_from_slice(
            &poly_bits_le[self.poly_coeff_bits..self.poly_coeff_bits + self.poly_inner_bits],
        );
        out.extend_from_slice(&cycle_bits_le[self.cycle_tile_bits()..]);
        out.extend_from_slice(&poly_bits_le[self.poly_tile_bits()..]);
        out
    }
}

pub(super) fn choose_packed_bit_layout<const D: usize, Cfg: CommitmentConfig>(
    log_k: usize,
    log_t: usize,
    log_packed: usize,
) -> PackedBitLayout {
    let alpha_bits = D.trailing_zeros() as usize;
    let addr_coeff_bits = log_k.min(alpha_bits);
    let addr_inner_bits = log_k - addr_coeff_bits;
    let lifted_coeff_bits = alpha_bits - addr_coeff_bits;
    assert!(
        lifted_coeff_bits <= log_t + log_packed,
        "packed layout cannot lift coeff bits from cycle/poly (lifted_coeff_bits={lifted_coeff_bits}, log_t={log_t}, log_packed={log_packed})"
    );
    let reduced_vars = addr_inner_bits + log_t + log_packed - lifted_coeff_bits;
    assert!(
        reduced_vars > 0,
        "packed layout expects at least one reduced variable"
    );

    let (optimal_m_vars, _) = optimal_m_r_split::<Cfg>(reduced_vars);
    let target_m_vars = optimal_m_vars.clamp(addr_inner_bits.max(1), reduced_vars);
    let target_cycle_poly_inner = target_m_vars - addr_inner_bits;
    let total_cycle_poly_tiled = target_cycle_poly_inner + lifted_coeff_bits;

    let desired_poly_tile = if log_packed == 0 {
        0
    } else {
        total_cycle_poly_tiled
            .min(log_packed)
            .clamp(1, PACKED_MAX_POLY_TILE_BITS)
    };

    let mut best_layout = None;
    let mut best_key = None;

    let min_cycle_coeff = lifted_coeff_bits.saturating_sub(log_packed);
    let max_cycle_coeff = lifted_coeff_bits.min(log_t);
    for cycle_coeff_bits in min_cycle_coeff..=max_cycle_coeff {
        let poly_coeff_bits = lifted_coeff_bits - cycle_coeff_bits;
        if poly_coeff_bits > log_packed {
            continue;
        }

        let remaining_cycle_bits = log_t - cycle_coeff_bits;
        let remaining_poly_bits = log_packed - poly_coeff_bits;
        let min_poly_inner = target_cycle_poly_inner.saturating_sub(remaining_cycle_bits);
        let max_poly_inner = target_cycle_poly_inner.min(remaining_poly_bits);
        for poly_inner_bits in min_poly_inner..=max_poly_inner {
            let cycle_inner_bits = target_cycle_poly_inner - poly_inner_bits;
            let layout = PackedBitLayout::new::<D>(
                log_k,
                log_t,
                log_packed,
                cycle_coeff_bits,
                poly_coeff_bits,
                cycle_inner_bits,
                poly_inner_bits,
            );
            let cycle_tile_bits = layout.cycle_tile_bits();
            let poly_tile_bits = layout.poly_tile_bits();
            let cycle_shortfall = if log_t >= PACKED_MIN_CYCLE_TILE_BITS {
                PACKED_MIN_CYCLE_TILE_BITS.saturating_sub(cycle_tile_bits)
            } else {
                0
            };
            let poly_shortfall = if log_packed > 0 && poly_tile_bits == 0 {
                1
            } else {
                0
            };
            let cycle_zero = if log_t > 0 && cycle_tile_bits == 0 {
                1
            } else {
                0
            };

            let key = (
                cycle_shortfall,
                poly_shortfall,
                desired_poly_tile.abs_diff(poly_tile_bits),
                cycle_zero,
                usize::MAX - cycle_tile_bits,
                poly_coeff_bits,
                poly_tile_bits,
            );

            if best_key.is_none_or(|best| key < best) {
                best_key = Some(key);
                best_layout = Some(layout);
            }
        }
    }

    best_layout.expect("packed layout candidate search produced no layout")
}
