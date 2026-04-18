//! Word-oriented memop overrides plus trace helpers for the
//! `stateless-evm` guest.
//!
//! # Why this exists
//!
//! Jolt's RAM model is 8-byte addressed: every sub-word load or store
//! (`LBU`, `SB`, `LW`, `SW`, etc.) expands into 7-14 virtual instructions
//! because the emulator has to read the containing 8-byte word, mask,
//! shift, and write back. `compiler_builtins`'s generic memops are good,
//! but they are not tuned for Jolt's cost model. In particular:
//!
//! 1. `memset` still uses mixed sub-word stores.
//! 2. `memcmp` is still a pure byte loop.
//! 3. `memcpy` / `memmove` have a good misaligned path, but we want to
//!    measure the exact size and alignment distribution in this guest and
//!    compare an explicit local port of that path against the baseline.
//!
//! The overrides here:
//!
//! 1. Replace `memcpy` / `memmove` with a local port of
//!    `compiler_builtins`' shifted-word algorithm.
//! 2. Replace `memset` with a byte-prefix + `u64`-stride fill + byte-tail
//!    sequence.
//! 3. Replace `memcmp` with a shared-alignment fast path: compare a short
//!    byte prefix until aligned, then compare full words.
//! 4. Collect size/alignment telemetry for `memcpy` / `memmove` so the
//!    host can explain what the remaining generic copy traffic looks like.
//! 5. Only define the C-ABI symbols when `target_arch = "riscv64"` so
//!    host builds still link against the system libc.
//!
//! Callers that know their pointers are word-aligned (e.g. after viewing
//! a `[U256]` stack slot as `[[u64; 4]]`) can bypass `memcpy` entirely
//! by using the safe helpers in the `safe` module.

#![no_std]

#[cfg(any(test, target_arch = "riscv64"))]
mod riscv_overrides;

pub mod safe;

/// Size of a Jolt-native word in bytes.
pub const WORD_BYTES: usize = 8;

/// Guest-side summary of memcpy / memmove call distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawMemopsTraceStats {
    pub memcpy_calls: u64,
    pub memcpy_bytes: u64,
    pub memmove_calls: u64,
    pub memmove_bytes: u64,
    pub memcpy_byte_only_calls: u64,
    pub memcpy_aligned_word_calls: u64,
    pub memcpy_misaligned_word_calls: u64,
    pub memmove_forward_byte_only_calls: u64,
    pub memmove_forward_aligned_word_calls: u64,
    pub memmove_forward_misaligned_word_calls: u64,
    pub memmove_backward_byte_only_calls: u64,
    pub memmove_backward_aligned_word_calls: u64,
    pub memmove_backward_misaligned_word_calls: u64,
    /// Call-count histogram of memcpy lengths:
    /// `[0..16), [16..32), [32..64), [64..128), [128..256), [256..512),
    /// [512..1024), [1024..inf)`.
    pub memcpy_size_hist: [u64; 8],
    /// Histogram of `src % 8`.
    pub memcpy_src_align_hist: [u64; 8],
    /// Histogram of `dst % 8`.
    pub memcpy_dst_align_hist: [u64; 8],
    /// Histogram of `(src ^ dst) & 7`, which determines whether a shifted
    /// word path is needed after aligning `dst`.
    pub memcpy_align_diff_hist: [u64; 8],
}

impl RawMemopsTraceStats {
    pub const ZERO: Self = Self {
        memcpy_calls: 0,
        memcpy_bytes: 0,
        memmove_calls: 0,
        memmove_bytes: 0,
        memcpy_byte_only_calls: 0,
        memcpy_aligned_word_calls: 0,
        memcpy_misaligned_word_calls: 0,
        memmove_forward_byte_only_calls: 0,
        memmove_forward_aligned_word_calls: 0,
        memmove_forward_misaligned_word_calls: 0,
        memmove_backward_byte_only_calls: 0,
        memmove_backward_aligned_word_calls: 0,
        memmove_backward_misaligned_word_calls: 0,
        memcpy_size_hist: [0; 8],
        memcpy_src_align_hist: [0; 8],
        memcpy_dst_align_hist: [0; 8],
        memcpy_align_diff_hist: [0; 8],
    };
}

impl Default for RawMemopsTraceStats {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Pull the C-ABI overrides into the final binary. On non-`riscv64` targets
/// this is a no-op so the helper can be called unconditionally from the
/// guest entry point.
#[inline(always)]
pub fn link_overrides() {
    #[cfg(target_arch = "riscv64")]
    riscv_overrides::link_overrides();
}

/// Reset the memcpy / memmove trace counters.
#[inline(always)]
pub fn reset_trace_stats() {
    #[cfg(target_arch = "riscv64")]
    riscv_overrides::reset_trace_stats();
}

/// Snapshot the memcpy / memmove trace counters.
#[inline(always)]
pub fn snapshot_trace_stats() -> RawMemopsTraceStats {
    #[cfg(target_arch = "riscv64")]
    {
        return riscv_overrides::snapshot_trace_stats();
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        RawMemopsTraceStats::ZERO
    }
}
