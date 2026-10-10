//! Views of caller-owned bucket arrays, indexed by position then value.
//!
//! A nibble position occupies 16 consecutive elements; a byte position occupies
//! 256. A set for one selector value and `words` word slots occupies
//! `words * 16 * 16` nibble elements (or `words * 8 * 256` byte elements).
//! Several sets are concatenated in selector order, then word-slot order, then
//! position order. A caller splits its scratch array at those set boundaries
//! before constructing the views. The column pass uses 32 byte positions,
//! exactly 8,192 elements. Digit histograms occupy separate contiguous ranges.
//! Array merging is owned by [`super::pool::ScratchPool::merge`].

use jolt_field::F128;
use thiserror::Error;

/// Invalid bucket geometry or an index outside a checked view.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BucketError {
    /// Storage must contain a whole number of positions.
    #[error("bucket length {len} is not a multiple of {entries}")]
    Length { len: usize, entries: usize },
    /// A position is outside the view's position count.
    #[error("bucket position {position} is outside {positions} positions")]
    Position { position: usize, positions: usize },
    /// A value is outside the position's value domain.
    #[error("bucket value {value} is outside {bound} values")]
    Value { value: usize, bound: usize },
    /// A requested bit is outside the value's width.
    #[error("bucket bit {bit} is outside width {bits}")]
    Bit { bit: usize, bits: usize },
    /// The histogram's power-of-two domain cannot be represented.
    #[error("digit width {bits} cannot be represented")]
    Width { bits: usize },
    /// Histogram storage differs from its digit domain.
    #[error("histogram length {len} differs from digit bound {bound}")]
    HistogramLength { len: usize, bound: usize },
}

macro_rules! buckets {
    ($name:ident, $bits:literal, $entries:literal, $doc:literal) => {
        #[doc = $doc]
        pub struct $name<'a> {
            positions: &'a mut [[F128; $entries]],
        }

        impl<'a> $name<'a> {
            /// Borrows storage containing a whole number of positions.
            /// Contents are retained; the caller supplies the initial sums.
            pub fn new(storage: &'a mut [F128]) -> Result<Self, BucketError> {
                let len = storage.len();
                let (positions, remainder) = storage.as_chunks_mut::<$entries>();
                if !remainder.is_empty() {
                    return Err(BucketError::Length {
                        len,
                        entries: $entries,
                    });
                }
                Ok(Self { positions })
            }

            /// Number of positions in this view, including zero for empty storage.
            pub fn positions(&self) -> usize {
                self.positions.len()
            }

            /// Adds `e` to the specified value's sum with one load-XOR-store.
            /// Checks both indices before changing storage.
            #[inline]
            pub fn xor(
                &mut self,
                position: usize,
                value: usize,
                e: F128,
            ) -> Result<(), BucketError> {
                let positions = self.positions.len();
                let bucket = self
                    .positions
                    .get_mut(position)
                    .ok_or(BucketError::Position {
                        position,
                        positions,
                    })?;
                let entry = bucket.get_mut(value).ok_or(BucketError::Value {
                    value,
                    bound: $entries,
                })?;
                *entry += e;
                Ok(())
            }

            /// Returns the sum for each bit of one position: bit `i` is the XOR
            /// of the buckets whose value has bit `i` set.
            pub fn bits(&self, position: usize) -> Result<[F128; $bits], BucketError> {
                let bucket = self.positions.get(position).ok_or(BucketError::Position {
                    position,
                    positions: self.positions.len(),
                })?;
                Ok(std::array::from_fn(|bit| {
                    bucket
                        .iter()
                        .enumerate()
                        .filter(|(value, _)| value & (1 << bit) != 0)
                        .fold(F128::from_raw(0), |sum, (_, &e)| sum + e)
                }))
            }

            /// Returns one bit's sum, rejecting a bit outside the value width.
            pub fn bit(&self, position: usize, bit: usize) -> Result<F128, BucketError> {
                if bit >= $bits {
                    return Err(BucketError::Bit { bit, bits: $bits });
                }
                Ok(self.bits(position)?[bit])
            }

            /// XOR of every value bucket at a position, including value zero.
            pub fn total(&self, position: usize) -> Result<F128, BucketError> {
                let bucket = self.positions.get(position).ok_or(BucketError::Position {
                    position,
                    positions: self.positions.len(),
                })?;
                Ok(bucket
                    .iter()
                    .copied()
                    .fold(F128::from_raw(0), |sum, e| sum + e))
            }
        }
    };
}

buckets!(
    NibbleBuckets,
    4,
    16,
    "Position-major nibble sums in borrowed scratch, 16 elements per position."
);
buckets!(
    ByteBuckets,
    8,
    256,
    "Position-major byte sums in borrowed scratch, 256 elements per position."
);

/// Per-digit sums in borrowed scratch with exactly `2^bits` entries.
pub struct DigitHistogram<'a> {
    sums: &'a mut [F128],
}

impl<'a> DigitHistogram<'a> {
    /// Checks the representable width and exact storage length; retains contents.
    pub fn new(storage: &'a mut [F128], bits: usize) -> Result<Self, BucketError> {
        let bound = 1_usize
            .checked_shl(u32::try_from(bits).unwrap_or(u32::MAX))
            .ok_or(BucketError::Width { bits })?;
        if storage.len() != bound {
            return Err(BucketError::HistogramLength {
                len: storage.len(),
                bound,
            });
        }
        Ok(Self { sums: storage })
    }

    /// Adds `e` to a digit's sum, checking the digit against the domain.
    #[inline]
    pub fn xor(&mut self, digit: usize, e: F128) -> Result<(), BucketError> {
        let bound = self.sums.len();
        let sum = self.sums.get_mut(digit).ok_or(BucketError::Value {
            value: digit,
            bound,
        })?;
        *sum += e;
        Ok(())
    }

    /// Sums in digit order, including digit zero.
    pub fn sums(&self) -> &[F128] {
        self.sums
    }
}
