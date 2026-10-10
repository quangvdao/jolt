use crate::packed::buckets::ByteBuckets;
use crate::packed::pool::ScratchPool;
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use jolt_field::F128;
use rayon::prelude::*;
use thiserror::Error;

const BUCKETS: usize = 32 * 256;

/// Rejected packed-row dimensions or cycle-point length.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ColumnPassError {
    /// The rows must span a nonempty Boolean cube.
    #[error("row count {rows} is not a positive power of two")]
    Rows { rows: usize },
    /// Coordinate `i` belongs to bit `i` of the cycle index.
    #[error("cycle point has {actual} coordinates, expected {expected}")]
    PointLength { expected: usize, actual: usize },
}

struct Prepared {
    chunks: CycleChunks,
    low: Vec<F128>,
    high: Vec<F128>,
    scratch: ScratchPool,
}

impl Prepared {
    #[expect(
        clippy::expect_used,
        reason = "a slice's power-of-two length gives representable geometry; the fixed 8192-element scratch length fits"
    )]
    fn new(rows: &[[u64; 4]], point: &[F128]) -> Result<Self, ColumnPassError> {
        if !rows.len().is_power_of_two() {
            return Err(ColumnPassError::Rows { rows: rows.len() });
        }
        let expected = rows.len().ilog2() as usize;
        if point.len() != expected {
            return Err(ColumnPassError::PointLength {
                expected,
                actual: point.len(),
            });
        }
        let chunks = CycleChunks::new(expected, 0).expect("checked cycle exponent");
        let (low, high) = chunks.split_point(point).expect("checked cycle point");
        Ok(Self {
            chunks,
            low: eq_table(low, None),
            high: eq_table(high, None),
            scratch: ScratchPool::new(BUCKETS).expect("fixed scratch size"),
        })
    }

    #[expect(
        clippy::expect_used,
        reason = "each chunk holds one sequential loan in the current pool; the fixed scratch length contains whole byte positions"
    )]
    fn pass(&self, rows: &[[u64; 4]]) {
        let block_len = self.chunks.block_len();
        let blocks_per_chunk = self.chunks.chunk_len() / block_len;
        rows.par_chunks(self.chunks.chunk_len())
            .zip(self.high.par_chunks(blocks_per_chunk))
            .for_each(|(rows, high)| {
                let mut scratch = self.scratch.take().expect("one loan per worker");
                let mut buckets = ByteBuckets::new(&mut scratch).expect("fixed bucket layout");
                for (block, &high) in rows.chunks_exact(block_len).zip(high) {
                    for (row, &low) in block.iter().zip(&self.low) {
                        let e = low * high;
                        let positions: &mut [[F128; 256]; 32] = buckets
                            .positions_mut()
                            .try_into()
                            .expect("fixed 32-position layout");
                        for (&word, positions) in row.iter().zip(positions.as_chunks_mut::<8>().0) {
                            for (value, bucket) in word.to_le_bytes().into_iter().zip(positions) {
                                bucket[usize::from(value)] += e;
                            }
                        }
                    }
                }
            });
    }

    #[expect(
        clippy::expect_used,
        reason = "all chunk loans have returned before the exclusive scratch merge"
    )]
    fn merge(&self) -> Vec<F128> {
        self.scratch.merge().expect("no outstanding scratch loans")
    }
}

#[expect(
    clippy::expect_used,
    reason = "only the merged fixed 32-position scratch layout is passed to this private readout"
)]
fn read_columns(sums: &mut [F128]) -> [F128; 256] {
    let buckets = ByteBuckets::new(sums).expect("fixed bucket layout");
    let mut columns = [F128::from_raw(0); 256];
    for (position, out) in columns.chunks_exact_mut(8).enumerate() {
        out.copy_from_slice(&buckets.bits(position).expect("position below 32"));
    }
    columns
}

/// Return `C[y] = Σ_j eq(r, j) Bits[y, j]` for all 256 packed columns.
///
/// `rows` must have a positive power-of-two length and `r` exactly its base-two
/// logarithm in coordinates, low variable first. Both conditions are checked.
/// The pass borrows the rows and retains no scratch after returning. It uses
/// two half equality tables and at most 128 KiB of byte buckets per Rayon worker.
/// There is no additional honest-input condition required of the caller, not
/// checked, or detected by the verifier.
pub fn column_pass(rows: &[[u64; 4]], r: &[F128]) -> Result<[F128; 256], ColumnPassError> {
    let prepared = Prepared::new(rows, r)?;
    prepared.pass(rows);
    Ok(read_columns(&mut prepared.merge()))
}
