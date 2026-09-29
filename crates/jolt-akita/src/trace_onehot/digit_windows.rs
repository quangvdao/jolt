//! Output-stationary commit accumulation for K<D trace one-hot rings.
//!
//! A K<D ring packs `D/K` trace rows, so each column adds several shifts of
//! the same `A` entry into one destination. Loading the entries of a chunk of
//! consecutive rings as negacyclic windows lets each destination tile sum the
//! shifts of the whole chunk in registers and touch memory once per chunk,
//! instead of once per shift. Chunks span enough rings for about
//! [`CHUNK_SHIFTS`] shifts per tile at any `D/K`: one ring at K=16 and D=512,
//! sixteen at K=256. Each window is split into 32-bit digit planes, and a pass
//! reads one plane of the chunk, so the bytes a pass reuses across columns
//! stay a quarter of the chunk's windows.

use akita_algebra::CyclotomicRing;
use jolt_field::{Fp128x8i32, Unreduced, Zero};

use super::traversal::row_is_committed;
use crate::AkitaField;

/// 32-bit digit planes of a canonical 128-bit value.
pub(super) const DIGIT_PLANES: usize = 4;
/// Coefficients per register tile. Sixteen coefficients of one plane are 32
/// `u32` sums, half of baseline x86-64's SSE2 register file.
const TILE: usize = 16;
/// Shifts per destination tile per chunk, enough that summing them outweighs
/// the tile's load and store.
const CHUNK_SHIFTS: usize = 32;

/// Two little-endian 16-bit digits: one plane of a canonical value.
type DigitPair = [u16; 2];

/// One destination ring element as unreduced `i32` lanes, plane-major:
/// entry `[q][j]` holds lanes `2q, 2q + 1` of coefficient `j`'s
/// [`Fp128x8i32`].
pub(super) type DigitAccumulator<const D: usize> = [[[i32; 2]; D]; DIGIT_PLANES];

/// Consecutive rings a chunk spans at `rows_per_ring` rows per ring.
pub(super) fn chunk_rings(rows_per_ring: usize) -> usize {
    (CHUNK_SHIFTS / rows_per_ring).max(1)
}

/// Every negacyclic shift of a chunk of `A` entries as 16-bit digit pairs.
///
/// Window `w` holds `[-a_0, …, -a_{D-1}, a_0, …, a_{D-1}]` for the chunk's
/// `w`-th entry `a`, so coefficient `j` of `a · X^k` is window entry
/// `D + j - k` for every `k < D`. Plane `q` stores digits `2q, 2q + 1` of
/// every window, window after window. The digits are the non-negative
/// [`Fp128x8i32`] lanes of each canonical value, so a shift adds a value below
/// `2^16` to each destination lane. At most `MAX_WIDE_ACCUMULATIONS` shifts per
/// destination between flushes keep every lane inside `reduce_wide`'s `i32`
/// range.
pub(super) struct DigitWindows<const D: usize> {
    planes: [Vec<DigitPair>; DIGIT_PLANES],
    negated: Vec<AkitaField>,
}

impl<const D: usize> DigitWindows<D> {
    pub(super) fn new(capacity: usize) -> Self {
        const { assert!(D.is_multiple_of(TILE)) };
        Self {
            planes: std::array::from_fn(|_| vec![[0; 2]; capacity * 2 * D]),
            negated: vec![AkitaField::zero(); D],
        }
    }

    /// Replaces window `window` with the shifts of `src`.
    pub(super) fn load(&mut self, window: usize, src: &CyclotomicRing<AkitaField, D>) {
        for (negated, &value) in self.negated.iter_mut().zip(&src.coeffs) {
            *negated = -value;
        }
        let [p0, p1, p2, p3] = self
            .planes
            .each_mut()
            .map(|plane| plane[window * 2 * D..][..2 * D].split_at_mut(D));
        fill_planes(p0.0, p1.0, p2.0, p3.0, &self.negated);
        fill_planes(p0.1, p1.1, p2.1, p3.1, &src.coeffs);
    }

    /// Adds `a_w · X^k` into accumulator `i` for every window offset of
    /// column `i` in `shifts`.
    pub(super) fn accumulate(
        &self,
        accumulators: &mut [DigitAccumulator<D>],
        shifts: &ChunkShifts,
    ) {
        // Plane-outer passes reuse one plane of the chunk across every column.
        for (plane_index, plane) in self.planes.iter().map(Vec::as_slice).enumerate() {
            for (column, accumulator) in accumulators.iter_mut().enumerate() {
                let offsets = shifts.column(column);
                if offsets.is_empty() {
                    continue;
                }
                for (tile, out) in accumulator[plane_index].chunks_exact_mut(TILE).enumerate() {
                    let mut sums = [[0u32; 2]; TILE];
                    for &offset in offsets {
                        let start = offset as usize + tile * TILE;
                        for (sum, digits) in sums.iter_mut().zip(&plane[start..start + TILE]) {
                            for (sum, &digit) in sum.iter_mut().zip(digits) {
                                *sum += u32::from(digit);
                            }
                        }
                    }
                    for (out, sum) in out.iter_mut().zip(&sums) {
                        for (lane, &sum) in out.iter_mut().zip(sum) {
                            *lane += sum as i32;
                        }
                    }
                }
            }
        }
    }
}

/// Splits `values` into their digit planes. Zipping the four plane slices
/// keeps the loop free of bounds checks and within the vectorizer's
/// alias-check budget.
fn fill_planes(
    p0: &mut [DigitPair],
    p1: &mut [DigitPair],
    p2: &mut [DigitPair],
    p3: &mut [DigitPair],
    values: &[AkitaField],
) {
    // Splitting the canonical limbs into 32-bit words, rather than the value
    // into 16-bit digits, is what the vectorizer turns into lane shuffles.
    let pair = |word: u64| [word as u16, (word >> 16) as u16];
    for ((((q0, q1), q2), q3), &value) in p0.iter_mut().zip(p1).zip(p2).zip(p3).zip(values) {
        let [lo, hi] = value.to_limbs();
        [*q0, *q1, *q2, *q3] = [pair(lo), pair(lo >> 32), pair(hi), pair(hi >> 32)];
    }
}

/// Window offsets of one chunk's committed rows, grouped by column.
pub(super) struct ChunkShifts {
    offsets: Vec<u32>,
    lens: Vec<usize>,
    max_per_column: usize,
}

impl ChunkShifts {
    pub(super) fn new(num_columns: usize, max_per_column: usize) -> Self {
        Self {
            offsets: vec![0; num_columns * max_per_column],
            lens: vec![0; num_columns],
            max_per_column,
        }
    }

    /// Replaces the offsets with the committed rows of a chunk, row-major.
    /// Row `i` lies in window `i / (D/K)` at shift `(i mod D/K)·K + hot`.
    pub(super) fn fill<const D: usize>(
        &mut self,
        selected_rows: &[u8],
        committed_zero_masks: &[u64],
        one_hot_k: usize,
    ) {
        let rows_per_ring = D / one_hot_k;
        let num_columns = self.lens.len();
        debug_assert!(committed_zero_masks.len() <= self.max_per_column);
        self.lens.fill(0);
        for (row, (row_indices, &committed_zero_mask)) in selected_rows
            .chunks_exact(num_columns)
            .zip(committed_zero_masks)
            .enumerate()
        {
            let window = row / rows_per_ring;
            let row_base = window * 2 * D + D - row % rows_per_ring * one_hot_k;
            for (column, (&hot, len)) in row_indices.iter().zip(&mut self.lens).enumerate() {
                // Every row writes its offset; only committed rows keep it.
                self.offsets[column * self.max_per_column + *len] =
                    (row_base - usize::from(hot)) as u32;
                *len += usize::from(row_is_committed(hot, committed_zero_mask, column));
            }
        }
    }

    fn column(&self, column: usize) -> &[u32] {
        &self.offsets[column * self.max_per_column..][..self.lens[column]]
    }
}

/// Adds every accumulator into its reduced ring element and clears it.
///
/// Accumulators are A-row-major, `[a][column]`; `reduced` is column-major,
/// `[column][a]`.
pub(super) fn flush_digit_accumulators<const D: usize>(
    accumulators: &mut [DigitAccumulator<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
    num_columns: usize,
) {
    let n_a = accumulators.len() / num_columns;
    for (index, accumulator) in accumulators.iter_mut().enumerate() {
        let reduced = &mut reduced[index % num_columns * n_a + index / num_columns];
        for (coefficient_index, coefficient) in reduced.coeffs.iter_mut().enumerate() {
            let lanes = std::array::from_fn(|lane| {
                std::mem::take(&mut accumulator[lane / 2][coefficient_index][lane % 2])
            });
            *coefficient += AkitaField::reduce_wide(Fp128x8i32(lanes));
        }
    }
}
