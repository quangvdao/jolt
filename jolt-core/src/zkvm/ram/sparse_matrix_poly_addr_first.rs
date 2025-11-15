use allocative::Allocative;
use rayon::prelude::*;

use crate::field::{JoltField, OptimizedMul};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Matrix entry used for the address-first RAM twist.
///
/// This is similar to `sparse_matrix_poly::MatrixEntry`, but:
/// - `prev_val` / `next_val` are stored as field elements instead of `u64`,
///   since binding address variables produces arbitrary linear combinations.
/// - The same struct is used for both address-binding and cycle-binding phases.
#[derive(Allocative, Debug, PartialEq)]
pub struct MatrixEntryAddrFirst<F: JoltField> {
    /// Cycle index. Before binding any cycle variables, row ∈ [0, T).
    pub row: usize,
    /// Address index. Before binding any address variables, col ∈ [0, K).
    pub col: usize,
    /// Snapshot of the value at the previous time window boundary for this
    /// (possibly virtual) address, as a field element.
    pub prev_val: F,
    /// Snapshot of the value at the next time window boundary for this
    /// (possibly virtual) address, as a field element.
    pub next_val: F,
    /// Current Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// Current ra coefficient for this matrix entry.
    pub ra_coeff: F,
}

/// Sparse matrix used for the address-first RAM twist.
///
/// Conceptually this represents the same object as `SparseMatrixPolynomial`:
/// a sparse K × T matrix of `(ra, Val)` coefficients. The difference is that
/// `prev_val` / `next_val` are kept in the field, so that we can bind address
/// variables first (forming virtual addresses) and then reuse the existing
/// cycle-binding logic.
#[derive(Allocative, Debug, Default)]
pub struct SparseMatrixPolynomialAddrFirst<F: JoltField> {
    pub entries: Vec<MatrixEntryAddrFirst<F>>,
}

impl<F: JoltField> SparseMatrixPolynomialAddrFirst<F> {
    /// Build the sparse matrix from the RAM trace and memory layout.
    ///
    /// This mirrors `SparseMatrixPolynomial::new`, but initializes
    /// `prev_val` / `next_val` as field elements.
    pub fn new(trace: &[Cycle], memory_layout: &MemoryLayout) -> Self {
        let mut entries: Vec<_> = trace
            .par_iter()
            .enumerate()
            .filter_map(|(j, cycle)| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Write(write) => {
                        let addr = remap_address(write.address, memory_layout)?;
                        let pre = F::from_u64(write.pre_value);
                        let post = F::from_u64(write.post_value);
                        Some(MatrixEntryAddrFirst {
                            row: j,
                            col: addr as usize,
                            ra_coeff: F::one(),
                            val_coeff: pre,
                            prev_val: pre,
                            next_val: post,
                        })
                    }
                    RAMAccess::Read(read) => {
                        let addr = remap_address(read.address, memory_layout)?;
                        let val = F::from_u64(read.value);
                        Some(MatrixEntryAddrFirst {
                            row: j,
                            col: addr as usize,
                            ra_coeff: F::one(),
                            val_coeff: val,
                            prev_val: val,
                            next_val: val,
                        })
                    }
                    _ => None,
                }
            })
            .collect();

        // Ensure entries are sorted by (row, col) so that address- and
        // cycle-binding routines can treat each row as a sorted run by col.
        entries.par_sort_unstable_by_key(|e| (e.row, e.col));

        SparseMatrixPolynomialAddrFirst { entries }
    }

    /// Bind one address bit across the whole sparse matrix.
    ///
    /// This corresponds to the pseudocode `SparseMatrixAddrFirst_BindAddressBit`
    /// in `twist-pseudocode.md`. Entries are assumed sorted by `(row, col)`.
    /// After this call, each row has half as many distinct address indices.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomialAddrFirst::bind_address_bit")]
    pub fn bind_address_bit(&mut self, r_addr: F::Challenge) {
        self.entries = self
            .entries
            .par_chunk_by(|x, y| x.row == y.row)
            .flat_map(|row_entries| {
                debug_assert!(!row_entries.is_empty());
                let row = row_entries[0].row;
                let mut out: Vec<MatrixEntryAddrFirst<F>> =
                    Vec::with_capacity(row_entries.len());

                let mut i = 0;
                while i < row_entries.len() {
                    let col = row_entries[i].col;
                    let base = col / 2;
                    let even_col = base * 2;
                    let odd_col = even_col + 1;

                    let mut even: Option<&MatrixEntryAddrFirst<F>> = None;
                    let mut odd: Option<&MatrixEntryAddrFirst<F>> = None;

                    if col == even_col {
                        even = Some(&row_entries[i]);
                        i += 1;
                        if i < row_entries.len() && row_entries[i].col == odd_col {
                            odd = Some(&row_entries[i]);
                            i += 1;
                        }
                    } else {
                        // First child we see is the odd address.
                        debug_assert_eq!(col, odd_col);
                        odd = Some(&row_entries[i]);
                        i += 1;
                    }

                    let parent =
                        Self::bind_address_pair(even, odd, r_addr, row, base);
                    out.push(parent);
                }

                out
            })
            .collect();
    }

    fn bind_address_pair(
        even: Option<&MatrixEntryAddrFirst<F>>,
        odd: Option<&MatrixEntryAddrFirst<F>>,
        r_addr: F::Challenge,
        row: usize,
        col_parent: usize,
    ) -> MatrixEntryAddrFirst<F> {
        match (even, odd) {
            (Some(e), Some(o)) => {
                // Both children present: full interpolation in the address bit.
                MatrixEntryAddrFirst {
                    row,
                    col: col_parent,
                    ra_coeff: e.ra_coeff + r_addr * (o.ra_coeff - e.ra_coeff),
                    val_coeff: e.val_coeff + r_addr.mul_0_optimized(o.val_coeff - e.val_coeff),
                    prev_val: e.prev_val + r_addr.mul_0_optimized(o.prev_val - e.prev_val),
                    next_val: e.next_val + r_addr.mul_0_optimized(o.next_val - e.next_val),
                }
            }
            (Some(e), None) => {
                // Only even side present: odd side has ra = 0 for all cycles.
                // We can safely reuse even's snapshots; Val on the missing side
                // never contributes to ra * (Val + gamma * (inc + Val)).
                MatrixEntryAddrFirst {
                    row,
                    col: col_parent,
                    ra_coeff: (F::one() - r_addr) * e.ra_coeff,
                    val_coeff: e.val_coeff,
                    prev_val: e.prev_val,
                    next_val: e.next_val,
                }
            }
            (None, Some(o)) => {
                // Only odd side present, symmetric case.
                MatrixEntryAddrFirst {
                    row,
                    col: col_parent,
                    ra_coeff: r_addr * o.ra_coeff,
                    val_coeff: o.val_coeff,
                    prev_val: o.prev_val,
                    next_val: o.next_val,
                }
            }
            (None, None) => unreachable!("bind_address_pair called with no children"),
        }
    }

    /// Materialize dense `ra(j)` and `Val(j)` polynomials over cycles after
    /// all address variables have been bound.
    ///
    /// After `log2(K)` address rounds, each row corresponds to a single
    /// "virtual address" obtained by folding all concrete addresses into one
    /// linear combination. At that point the RAM read–write sumcheck only
    /// needs, for each cycle `j`,
    ///
    ///   ra'(j) and Val'(j)
    ///
    /// to define the univariate polynomial in the remaining cycle variables.
    ///
    /// This routine builds dense vectors of length `T` indexed by cycle:
    /// - rows that never had a RAM access keep `ra'(j) = 0`;
    /// - `Val'(j)` is set to 0 there as well, which is sound for the
    ///   polynomial ra'(j) * (Val'(j) + gamma * (inc(j) + Val'(j))) since the
    ///   factor ra'(j) is zero.
    #[tracing::instrument(
        skip_all,
        name = "SparseMatrixPolynomialAddrFirst::materialize_over_cycles"
    )]
    pub fn materialize_over_cycles(
        self,
        T: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let mut ra = vec![F::zero(); T];
        let mut val = vec![F::zero(); T];

        for entry in self.entries {
            debug_assert!(entry.row < T);
            ra[entry.row] = entry.ra_coeff;
            val[entry.row] = entry.val_coeff;
        }

        (ra.into(), val.into())
    }
}


