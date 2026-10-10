//! Linear maps from packed bits to field elements through one-bit, two-bit,
//! nibble or byte lookup tables, including compact maps in caller-provided arenas.
//!
//! A caller folds any scalar multiplying the map into the supplied weights
//! before construction. Lifting uses lookups and field addition only.

use jolt_field::F128;
use thiserror::Error;

/// Invalid weights, compact table dimensions or arena layouts for a packed-bit lift.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiftError {
    /// More than 64 significant-bit weights were supplied.
    #[error("a word has at most 64 significant bits, received {count} weights")]
    WeightCount {
        /// The supplied number of weights.
        count: usize,
    },
    /// A compact lookup width must be one, two, four, or eight bits.
    #[error("unsupported compact lift width {bits}")]
    Width { bits: usize },
    /// A compact table must contain two, four, sixteen or 256 entries.
    #[error("unsupported compact table length {entries}")]
    TableLength { entries: usize },
    /// A lookup table received more weights than its index width.
    #[error("compact table accepts {capacity} weights, received {count}")]
    TableWeightCount { capacity: usize, count: usize },
    /// A static view disagrees with the arena's table layout.
    #[error("compact lift view does not match its arena layout")]
    Layout,
}

/// The map `word ↦ Σ_i word[i]·weights[i]` over all 64 bits of a word.
///
/// Eight independent 256-entry tables occupy 32 KiB. Each lift reads one entry
/// from each byte table and combines them with seven field additions.
#[derive(Debug)]
pub struct WordLift {
    tables: [[F128; 256]; 8],
}

impl WordLift {
    /// Builds all byte tables from 64 low-bit-first weights.
    ///
    /// The array type checks the weight count. Every field element is valid,
    /// so this constructor has no rejecting input. To scale the map, supply
    /// weights already multiplied by the scalar.
    pub fn new(weights: &[F128; 64]) -> Self {
        let tables = std::array::from_fn(|byte| table(&weights[8 * byte..8 * byte + 8]));
        Self { tables }
    }

    /// Returns the sum of the weights of all set bits, without multiplication.
    #[inline]
    pub fn lift(&self, word: u64) -> F128 {
        let [a, b, c, d, e, f, g, h] = word.to_le_bytes();
        self.tables[0][usize::from(a)]
            + self.tables[1][usize::from(b)]
            + self.tables[2][usize::from(c)]
            + self.tables[3][usize::from(d)]
            + self.tables[4][usize::from(e)]
            + self.tables[5][usize::from(f)]
            + self.tables[6][usize::from(g)]
            + self.tables[7][usize::from(h)]
    }
}

/// A lift of the low significant bits of a word compacted by `gather`.
///
/// For `b` supplied weights there are `ceil(b / 4)` independent 16-entry tables.
/// Table `p` reads nibble `p`, bits `4p..4p + 4`, of the compacted word. Bits at
/// or above `b` contribute zero, including padding in its final nibble. A word
/// with one significant bit at offset zero in every `2^m`-bit window supplies
/// `b = 64 / 2^m` weights after compaction.
#[derive(Debug)]
pub struct NibbleLift {
    tables: Box<[[F128; 16]]>,
}

impl NibbleLift {
    /// Builds nibble tables for zero through 64 low-bit-first weights.
    ///
    /// Returns `LiftError::WeightCount` above 64 weights. An empty map always
    /// returns zero. Supply pre-scaled weights when a scalar multiplies the map.
    pub fn new(weights: &[F128]) -> Result<Self, LiftError> {
        if weights.len() > 64 {
            return Err(LiftError::WeightCount {
                count: weights.len(),
            });
        }
        let tables = weights.chunks(4).map(table).collect();
        Ok(Self { tables })
    }

    /// Returns `Σ_i word[i]·weights[i]` over the supplied significant bits.
    ///
    /// Higher bits are ignored. No allocation or multiplication occurs.
    #[inline]
    pub fn lift(&self, mut word: u64) -> F128 {
        let mut sum = F128::from_raw(0);
        for table in &self.tables {
            sum += table[(word & 15) as usize];
            word >>= 4;
        }
        sum
    }
}

fn table<const N: usize>(weights: &[F128]) -> [F128; N] {
    let mut entries = [F128::from_raw(0); N];
    for (bit, &weight) in weights.iter().enumerate() {
        let width = 1 << bit;
        let (low, high) = entries[..2 * width].split_at_mut(width);
        for (dest, &src) in high.iter_mut().zip(low.iter()) {
            *dest = src + weight;
        }
    }
    if weights.len() < N.ilog2() as usize {
        let width = 1 << weights.len();
        let (low, high) = entries.split_at_mut(width);
        for chunk in high.chunks_mut(width) {
            chunk.copy_from_slice(low);
        }
    }
    entries
}

#[derive(Clone, Copy)]
/// A range of lookup tables in an arena preserved by the caller. Appending
/// tables preserves existing lifts; replacing entries changes their map.
pub(crate) struct CompactLift {
    start: usize,
    len: usize,
    bits: usize,
}

impl CompactLift {
    pub(crate) fn new(
        weights: &[F128],
        bits: usize,
        arena: &mut Vec<F128>,
    ) -> Result<Self, LiftError> {
        if weights.len() > 64 {
            return Err(LiftError::WeightCount {
                count: weights.len(),
            });
        }
        if !matches!(bits, 1 | 2 | 4 | 8) {
            return Err(LiftError::Width { bits });
        }
        let start = arena.len();
        for weights in weights.chunks(bits) {
            match bits {
                1 => arena.extend_from_slice(&table::<2>(weights)),
                2 => arena.extend_from_slice(&table::<4>(weights)),
                4 => arena.extend_from_slice(&table::<16>(weights)),
                _ => arena.extend_from_slice(&table::<256>(weights)),
            }
        }
        Ok(Self {
            start,
            len: arena.len() - start,
            bits,
        })
    }

    /// Checks width, table count and arena bounds before forming a static view.
    pub(crate) fn view<const N: usize, const TABLES: usize>(
        self,
        arena: &[F128],
    ) -> Result<CompactView<'_, N, TABLES>, LiftError> {
        if N != 1 << self.bits || Some(self.len) != N.checked_mul(TABLES) {
            return Err(LiftError::Layout);
        }
        let values = arena
            .get(self.start..self.start + self.len)
            .ok_or(LiftError::Layout)?;
        let tables = values
            .as_chunks::<N>()
            .0
            .try_into()
            .map_err(|_| LiftError::Layout)?;
        Ok(CompactView { tables })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CompactView<'a, const N: usize, const TABLES: usize> {
    tables: &'a [[F128; N]; TABLES],
}

impl<const N: usize, const TABLES: usize> CompactView<'_, N, TABLES> {
    #[inline]
    pub(crate) fn lift(self, mut word: u64) -> F128 {
        let mut sum = F128::from_raw(0);
        for table in self.tables {
            sum += table[(word & (N as u64 - 1)) as usize];
            word >>= N.ilog2();
        }
        sum
    }
}

pub(crate) fn compact_table<const N: usize>(weights: &[F128]) -> Result<[F128; N], LiftError> {
    if !matches!(N, 2 | 4 | 16 | 256) {
        return Err(LiftError::TableLength { entries: N });
    }
    if weights.len() > N.ilog2() as usize {
        return Err(LiftError::TableWeightCount {
            capacity: N.ilog2() as usize,
            count: weights.len(),
        });
    }
    Ok(table(weights))
}
