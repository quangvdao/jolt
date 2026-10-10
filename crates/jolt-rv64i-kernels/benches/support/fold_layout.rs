//! Bucket geometry shared by the fold calibration and packed-machinery passes.
//!
//! Variant word sets come first, then its eight digit/flag positions for every
//! selector, followed by Shift, Memory, Compare and Branch word sets. Word slots
//! contain sixteen nibble positions, each with sixteen adjacent field entries.

/// Field entries for one word's sixteen nibble positions.
pub const NIBBLE_WORD: usize = 16 * 16;
/// Field entries for one word's eight byte positions.
pub const BYTE_WORD: usize = 8 * 256;
/// Selector values of Variant, Shift, Memory, Compare and Branch.
pub const SELECTORS: [usize; 5] = [64, 512, 128, 512, 1];
/// Word slots per selector of the five shape ranges.
pub const WORD_SETS: [usize; 5] = [5, 1, 2, 3, 2];
/// Digit and combined-flag positions per Variant selector.
pub const METADATA_POSITIONS: usize = 8;

/// All Variant words precede metadata; subsequent shape ranges are contiguous.
#[derive(Clone, Copy)]
pub struct FoldLayout {
    byte_selectors: usize,
    offsets: [usize; 5],
    entries: usize,
}

impl FoldLayout {
    /// The caller selects at most 64 byte-bucketed Variant values.
    pub const fn new(byte_selectors: usize) -> Self {
        let metadata = byte_selectors * WORD_SETS[0] * BYTE_WORD
            + (SELECTORS[0] - byte_selectors) * WORD_SETS[0] * NIBBLE_WORD;
        let shift = metadata + SELECTORS[0] * METADATA_POSITIONS * 16;
        let memory = shift + SELECTORS[1] * WORD_SETS[1] * NIBBLE_WORD;
        let compare = memory + SELECTORS[2] * WORD_SETS[2] * NIBBLE_WORD;
        let branch = compare + SELECTORS[3] * WORD_SETS[3] * NIBBLE_WORD;
        Self {
            byte_selectors,
            offsets: [metadata, shift, memory, compare, branch],
            entries: branch + SELECTORS[4] * WORD_SETS[4] * NIBBLE_WORD,
        }
    }

    /// Total field entries, including every selector's words and metadata.
    pub const fn entries(self) -> usize {
        self.entries
    }

    /// Metadata, Shift, Memory, Compare and Branch starts, in field entries.
    pub const fn offsets(self) -> [usize; 5] {
        self.offsets
    }

    /// Word-set start for a selector checked by its source's digit width.
    #[inline]
    pub const fn variant_base(self, selector: usize) -> usize {
        if selector < self.byte_selectors {
            selector * WORD_SETS[0] * BYTE_WORD
        } else {
            self.byte_selectors * WORD_SETS[0] * BYTE_WORD
                + (selector - self.byte_selectors) * WORD_SETS[0] * NIBBLE_WORD
        }
    }

    /// Shape word-set start; the caller supplies a valid shape and selector.
    #[inline]
    pub fn shape_base(self, shape: usize, selector: usize) -> usize {
        if shape == 0 {
            self.variant_base(selector)
        } else {
            self.offsets[shape] + selector * WORD_SETS[shape] * NIBBLE_WORD
        }
    }

    /// Start of one Variant selector's seven digit and one flag positions.
    #[inline]
    pub const fn variant_metadata_base(self, selector: usize) -> usize {
        self.offsets[0] + selector * METADATA_POSITIONS * 16
    }
}
