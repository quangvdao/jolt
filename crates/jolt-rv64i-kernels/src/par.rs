//! Deterministic cycle geometry for split equality and parallel passes.
//!
//! Coordinates are low-bit-first. The original high half has `floor(log_t/2)`
//! variables; binding removes low variables first, then high variables. A block
//! fixes one high-half index and ranges over all remaining low-half indices.
//! Chunks contain whole blocks and target 4,096 entries, rounded up to a block
//! when a block is larger. Neither the split nor the chunk size reads a pool's
//! thread count. `round` is the number of cycle variables already removed.

use std::ops::Range;
use thiserror::Error;

/// Invalid cycle exponent, round or point length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ParError {
    #[error("cycle exponent {log_t} is not below {limit}")]
    LogSize { log_t: usize, limit: usize },
    #[error("round {round} exceeds cycle exponent {log_t}")]
    Round { log_t: usize, round: usize },
    #[error("point length {actual} differs from remaining dimension {expected}")]
    PointLength { expected: usize, actual: usize },
}

/// Split-equality block and chunk geometry after a fixed number of binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CycleChunks {
    low_bits: usize,
    high_bits: usize,
    len: usize,
    chunk_len: usize,
}

impl CycleChunks {
    /// Validate the exponent and round and choose chunks of whole high blocks.
    pub fn new(log_t: usize, round: usize) -> Result<Self, ParError> {
        if log_t >= usize::BITS as usize {
            return Err(ParError::LogSize {
                log_t,
                limit: usize::BITS as usize,
            });
        }
        if round > log_t {
            return Err(ParError::Round { log_t, round });
        }
        let remaining = log_t - round;
        let high_bits = (log_t / 2).min(remaining);
        let low_bits = remaining - high_bits;
        let len = 1 << remaining;
        let chunk_len = (1_usize << low_bits).max(4096).min(len);
        Ok(Self {
            low_bits,
            high_bits,
            len,
            chunk_len,
        })
    }

    /// Remaining low-half variable count, also the log of a block's length.
    pub fn low_bits(self) -> usize {
        self.low_bits
    }

    /// Remaining high-half variable count.
    pub fn high_bits(self) -> usize {
        self.high_bits
    }

    /// Number of remaining Boolean indices, including a scalar's one entry.
    pub fn len(self) -> usize {
        self.len
    }

    /// The Boolean domain always contains at least the empty assignment.
    pub fn is_empty(self) -> bool {
        false
    }

    /// Entries per chunk; divides the domain length and is a whole number of blocks.
    pub fn chunk_len(self) -> usize {
        self.chunk_len
    }

    /// Length of one complete low-half block for a fixed high-half index.
    pub fn block_len(self) -> usize {
        1 << self.low_bits
    }

    /// The low and high coordinate slices of a remaining low-first point.
    pub fn split_point<F>(self, point: &[F]) -> Result<(&[F], &[F]), ParError> {
        let expected = self.low_bits + self.high_bits;
        if point.len() != expected {
            return Err(ParError::PointLength {
                expected,
                actual: point.len(),
            });
        }
        Ok(point.split_at(self.low_bits))
    }

    /// Disjoint ranges covering the whole remaining cube in index order.
    pub fn ranges(self) -> impl ExactSizeIterator<Item = Range<usize>> {
        (0..self.len / self.chunk_len).map(move |chunk| {
            let start = chunk * self.chunk_len;
            start..start + self.chunk_len
        })
    }
}
