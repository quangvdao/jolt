//! Views of caller-owned bucket arrays, indexed by position then value.
//!
//! # Placement
//!
//! A store to one table and a load from another whose addresses agree in
//! their low 12 bits are treated by the processor as dependent until both
//! addresses are resolved. A pass that adds one weight to several tables at
//! the *same value* (the zero bytes of a small word above all) pays for
//! every such pair, so tables that one word set updates together are placed
//! off the 4,096-byte stride. [`BucketPlacement`] is the one owner of that
//! placement: one padding element after each 256 elements (4,096 bytes of
//! `F128`).
//!
//! - Byte positions stride by 257 elements. Two entries agree modulo 4,096
//!   bytes exactly when `Δposition + Δvalue = 0 (mod 256)`. The column pass
//!   has 32 positions and the largest bank of byte words 72, so equal values
//!   never agree within either.
//! - Nibble words stride by 257 elements, their 16 positions at `16 * p`.
//!   Two entries agree exactly when `Δword + 16 * Δposition + Δvalue = 0
//!   (mod 256)`. A set has at most nine words, so `|Δword| <= 8` and
//!   `|Δposition| <= 15`, and at equal value both differences are zero. Four
//!   padding elements would not do: `4 * Δword + 16 * Δposition = 0` at
//!   `(4, -1)`.
//!
//! The law covers the positions of the column pass and the words of one
//! selector's set, and `equal_value_tables_do_not_alias` checks exactly
//! that. It does not cover two sets that one cycle writes (one per shape),
//! nor a set against the digit and flag tables that follow it: a nibble
//! word takes one of the sixteen residue classes of its base modulo 16
//! elements, so only a layout that assigned those classes across every
//! shape at once could, and the fold has none. Unequal values can agree in
//! either form.
//!
//! The borrowed views below are dense value domains, excluding padding.
//! Digit histograms occupy separate contiguous ranges. Array merging is owned
//! by [`super::pool::ScratchPool::merge`].

use jolt_field::F128;
use thiserror::Error;

/// Placement of word buckets in scratch, excluding padding from value domains.
#[derive(Debug, Clone, Copy)]
pub enum BucketPlacement {
    Nibble,
    Byte,
}

impl BucketPlacement {
    /// Select the word encoding used by a fold selector value.
    pub const fn from_bytes(bytes: bool) -> Self {
        if bytes {
            Self::Byte
        } else {
            Self::Nibble
        }
    }

    /// Scratch elements occupied by one word, including block padding.
    pub const fn word_entries(self) -> usize {
        match self {
            Self::Nibble => Self::padded(NibbleBuckets::ELEMENTS_PER_WORD),
            Self::Byte => Self::padded(ByteBuckets::ELEMENTS_PER_WORD),
        }
    }

    /// Offset of a position within a word; byte positions also index whole passes.
    pub const fn position_offset(self, position: usize) -> usize {
        match self {
            Self::Nibble => position * NibbleBuckets::ENTRIES_PER_POSITION,
            Self::Byte => Self::padded(position * ByteBuckets::ENTRIES_PER_POSITION),
        }
    }

    const fn padded(elements: usize) -> usize {
        elements + elements / (4096 / std::mem::size_of::<F128>())
    }
}

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
            /// Number of value bits represented by one position.
            pub const BITS_PER_POSITION: usize = $bits;
            /// Number of value buckets in one position.
            pub const ENTRIES_PER_POSITION: usize = $entries;
            /// Number of positions needed for one 64-bit word.
            pub const POSITIONS_PER_WORD: usize = 64 / Self::BITS_PER_POSITION;
            /// Number of field elements needed for one 64-bit word.
            pub const ELEMENTS_PER_WORD: usize =
                Self::POSITIONS_PER_WORD * Self::ENTRIES_PER_POSITION;

            /// Returns every bit's XOR sum from a position in value order.
            /// Whether entries represent the claimed source is required of the
            /// caller, not checked here, detected by the verifier through the resulting claims.
            #[inline]
            pub fn position_bits(position: &[F128; $entries]) -> [F128; $bits] {
                std::array::from_fn(|bit| bit_sum(position, bit))
            }

            /// Returns one bit's XOR sum from a position in value order,
            /// rejecting a bit outside its width. Whether entries represent the
            /// claimed source is required of the caller, not checked here,
            /// detected by the verifier through the resulting claims.
            #[inline]
            pub fn position_bit(
                position: &[F128; $entries],
                bit: usize,
            ) -> Result<F128, BucketError> {
                if bit >= Self::BITS_PER_POSITION {
                    return Err(BucketError::Bit {
                        bit,
                        bits: Self::BITS_PER_POSITION,
                    });
                }
                Ok(bit_sum(position, bit))
            }

            /// XOR of all buckets in a read-only position, including value zero.
            /// Whether entries represent the claimed source is required of the
            /// caller, not checked here, detected by the verifier through the resulting claims.
            #[inline]
            pub fn position_total(position: &[F128; $entries]) -> F128 {
                position
                    .iter()
                    .copied()
                    .fold(F128::from_raw(0), |sum, e| sum + e)
            }
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

            /// Borrows whole position blocks, retaining their sums. Construction
            /// checks the storage length; the caller selects positions and uses
            /// values below the block's fixed entry count.
            #[inline]
            pub fn positions_mut(&mut self) -> &mut [[F128; $entries]] {
                self.positions
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
                Ok(Self::position_bits(bucket))
            }

            /// Returns one bit's sum, rejecting a bit outside the value width.
            pub fn bit(&self, position: usize, bit: usize) -> Result<F128, BucketError> {
                if bit >= $bits {
                    return Err(BucketError::Bit { bit, bits: $bits });
                }
                let bucket = self.positions.get(position).ok_or(BucketError::Position {
                    position,
                    positions: self.positions.len(),
                })?;
                Self::position_bit(bucket, bit)
            }

            /// XOR of every value bucket at a position, including value zero.
            pub fn total(&self, position: usize) -> Result<F128, BucketError> {
                let bucket = self.positions.get(position).ok_or(BucketError::Position {
                    position,
                    positions: self.positions.len(),
                })?;
                Ok(Self::position_total(bucket))
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
        Self::check_domain(storage.len(), bits)?;
        Ok(Self { sums: storage })
    }

    /// XOR of buckets whose digit has `bit` set, in a read-only `2^bits`
    /// domain. Checks the width, exact length and requested bit. Whether entries
    /// represent the claimed source is required of the caller, not checked here,
    /// detected by the verifier through the resulting claims.
    #[inline]
    pub fn sum_bit(storage: &[F128], bits: usize, bit: usize) -> Result<F128, BucketError> {
        Self::check_domain(storage.len(), bits)?;
        if bit >= bits {
            return Err(BucketError::Bit { bit, bits });
        }
        Ok(bit_sum(storage, bit))
    }

    fn check_domain(len: usize, bits: usize) -> Result<(), BucketError> {
        let bound = 1_usize
            .checked_shl(u32::try_from(bits).unwrap_or(u32::MAX))
            .ok_or(BucketError::Width { bits })?;
        if len != bound {
            return Err(BucketError::HistogramLength { len, bound });
        }
        Ok(())
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

fn bit_sum(bucket: &[F128], bit: usize) -> F128 {
    let half = 1 << bit;
    bucket
        .chunks_exact(2 * half)
        .flat_map(|period| &period[half..])
        .copied()
        .fold(F128::from_raw(0), |sum, e| sum + e)
}
