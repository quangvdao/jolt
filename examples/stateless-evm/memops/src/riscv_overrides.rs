//! Experimental memop overrides and telemetry for the `stateless-evm`
//! guest.
//!
//! The key experiment in this file is a local port of
//! `compiler_builtins`' shifted-word `memcpy` / `memmove` path, plus
//! call-distribution counters so we can see which copy sizes and
//! alignments actually dominate the mainnet trace.
//!
//! Unlike the first naive attempt, the copy path below does not give up
//! when `src` and `dst` have different 8-byte alignments. It aligns
//! `dst`, then reassembles aligned destination words from adjacent aligned
//! source words with shifts and ors.

#[cfg(target_arch = "riscv64")]
use crate::RawMemopsTraceStats;
use crate::WORD_BYTES;
#[cfg(target_arch = "riscv64")]
use core::cell::UnsafeCell;

type Word = u64;

const WORD_SIZE: usize = WORD_BYTES;
const WORD_MASK: usize = WORD_SIZE - 1;
const WORD_COPY_THRESHOLD: usize = if 2 * WORD_SIZE > 16 {
    2 * WORD_SIZE
} else {
    16
};
const WORD_COMPARE_THRESHOLD: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyPath {
    ByteOnly,
    AlignedWords,
    MisalignedWords,
}

#[cfg(target_arch = "riscv64")]
type MemcpyFn = unsafe extern "C" fn(*mut u8, *const u8, usize) -> *mut u8;
#[cfg(target_arch = "riscv64")]
type MemsetFn = unsafe extern "C" fn(*mut u8, i32, usize) -> *mut u8;
#[cfg(target_arch = "riscv64")]
type MemcmpFn = unsafe extern "C" fn(*const u8, *const u8, usize) -> i32;

#[cfg(target_arch = "riscv64")]
struct StatsCell(UnsafeCell<RawMemopsTraceStats>);

#[cfg(target_arch = "riscv64")]
// SAFETY: The stateless-evm guest executes single-threaded inside the Jolt
// emulator. We only mutate this cell through local helper functions while the
// guest is running, so treating it as `Sync` is sound for this target.
unsafe impl Sync for StatsCell {}

#[cfg(target_arch = "riscv64")]
static MEMOPS_TRACE_STATS: StatsCell = StatsCell(UnsafeCell::new(RawMemopsTraceStats::ZERO));

#[cfg(target_arch = "riscv64")]
#[used]
static KEEP_MEMCPY: MemcpyFn = memcpy;
#[cfg(target_arch = "riscv64")]
#[used]
static KEEP_MEMMOVE: MemcpyFn = memmove;
#[cfg(target_arch = "riscv64")]
#[used]
static KEEP_MEMSET: MemsetFn = memset;
#[cfg(target_arch = "riscv64")]
#[used]
static KEEP_MEMCMP: MemcmpFn = memcmp;

#[cfg(target_arch = "riscv64")]
const fn memcpy_size_bucket(n: usize) -> usize {
    if n < 16 {
        0
    } else if n < 32 {
        1
    } else if n < 64 {
        2
    } else if n < 128 {
        3
    } else if n < 256 {
        4
    } else if n < 512 {
        5
    } else if n < 1024 {
        6
    } else {
        7
    }
}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn with_stats_mut<R>(f: impl FnOnce(&mut RawMemopsTraceStats) -> R) -> R {
    // SAFETY: See the `Sync` justification on `StatsCell` above. We only
    // hand out one mutable reference at a time inside this helper.
    unsafe { f(&mut *MEMOPS_TRACE_STATS.0.get()) }
}

#[cfg(target_arch = "riscv64")]
pub fn reset_trace_stats() {
    with_stats_mut(|stats| *stats = RawMemopsTraceStats::ZERO);
}

#[cfg(target_arch = "riscv64")]
pub fn snapshot_trace_stats() -> RawMemopsTraceStats {
    // SAFETY: Reading a copy of the stats is safe for the same reason as
    // `with_stats_mut`; the guest is single-threaded.
    unsafe { *MEMOPS_TRACE_STATS.0.get() }
}

#[cfg(target_arch = "riscv64")]
fn record_memcpy_call(dst: *mut u8, src: *const u8, n: usize, path: CopyPath) {
    with_stats_mut(|stats| {
        stats.memcpy_calls += 1;
        stats.memcpy_bytes += n as u64;
        stats.memcpy_size_hist[memcpy_size_bucket(n)] += 1;
        stats.memcpy_src_align_hist[(src as usize) & WORD_MASK] += 1;
        stats.memcpy_dst_align_hist[(dst as usize) & WORD_MASK] += 1;
        stats.memcpy_align_diff_hist[((src as usize) ^ (dst as usize)) & WORD_MASK] += 1;
        match path {
            CopyPath::ByteOnly => stats.memcpy_byte_only_calls += 1,
            CopyPath::AlignedWords => stats.memcpy_aligned_word_calls += 1,
            CopyPath::MisalignedWords => stats.memcpy_misaligned_word_calls += 1,
        }
    });
}

#[cfg(target_arch = "riscv64")]
fn record_memmove_call(n: usize, backward: bool, path: CopyPath) {
    with_stats_mut(|stats| {
        stats.memmove_calls += 1;
        stats.memmove_bytes += n as u64;
        match (backward, path) {
            (false, CopyPath::ByteOnly) => stats.memmove_forward_byte_only_calls += 1,
            (false, CopyPath::AlignedWords) => stats.memmove_forward_aligned_word_calls += 1,
            (false, CopyPath::MisalignedWords) => stats.memmove_forward_misaligned_word_calls += 1,
            (true, CopyPath::ByteOnly) => stats.memmove_backward_byte_only_calls += 1,
            (true, CopyPath::AlignedWords) => stats.memmove_backward_aligned_word_calls += 1,
            (true, CopyPath::MisalignedWords) => stats.memmove_backward_misaligned_word_calls += 1,
        }
    });
}

/// Call once from the guest crate to guarantee this compilation unit is
/// pulled in by the linker. The body is intentionally trivial.
#[cfg(target_arch = "riscv64")]
#[inline(never)]
pub fn link_overrides() {
    core::hint::black_box((&KEEP_MEMCPY, &KEEP_MEMMOVE, &KEEP_MEMSET, &KEEP_MEMCMP));
}

#[inline(always)]
unsafe fn copy_forward_bytes(mut dest: *mut u8, mut src: *const u8, n: usize) {
    let dest_end = dest.wrapping_add(n);
    while dest < dest_end {
        *dest = *src;
        dest = dest.wrapping_add(1);
        src = src.wrapping_add(1);
    }
}

#[inline(always)]
unsafe fn copy_forward_aligned_words(dest: *mut u8, src: *const u8, n: usize) {
    let mut dest_word = dest.cast::<Word>();
    let mut src_word = src.cast::<Word>();
    let dest_end = dest.wrapping_add(n).cast::<Word>();
    while dest_word < dest_end {
        *dest_word = *src_word;
        dest_word = dest_word.wrapping_add(1);
        src_word = src_word.wrapping_add(1);
    }
}

#[inline(always)]
unsafe fn load_chunk_aligned<T: Copy>(
    src: *const Word,
    dst: *mut Word,
    load_sz: usize,
    offset: usize,
) -> usize {
    let chunk_sz = core::mem::size_of::<T>();
    if (load_sz & chunk_sz) != 0 {
        *dst.wrapping_byte_add(offset).cast::<T>() = *src.wrapping_byte_add(offset).cast::<T>();
        offset | chunk_sz
    } else {
        offset
    }
}

#[inline(always)]
unsafe fn load_aligned_partial(src: *const Word, load_sz: usize) -> Word {
    debug_assert!(load_sz < WORD_SIZE);
    let mut offset = 0;
    let mut out = 0u64;
    offset = load_chunk_aligned::<u32>(src, &raw mut out, load_sz, offset);
    offset = load_chunk_aligned::<u16>(src, &raw mut out, load_sz, offset);
    offset = load_chunk_aligned::<u8>(src, &raw mut out, load_sz, offset);
    debug_assert!(offset == load_sz);
    out
}

#[inline(always)]
unsafe fn load_aligned_end_partial(src: *const Word, load_sz: usize) -> Word {
    debug_assert!(load_sz < WORD_SIZE);
    let mut offset = 0;
    let mut out = 0u64;
    let src_shifted = src.wrapping_byte_add(WORD_SIZE - load_sz);
    let out_shifted = (&raw mut out).wrapping_byte_add(WORD_SIZE - load_sz);
    offset = load_chunk_aligned::<u8>(src_shifted, out_shifted.cast::<Word>(), load_sz, offset);
    offset = load_chunk_aligned::<u16>(src_shifted, out_shifted.cast::<Word>(), load_sz, offset);
    offset = load_chunk_aligned::<u32>(src_shifted, out_shifted.cast::<Word>(), load_sz, offset);
    debug_assert!(offset == load_sz);
    out
}

#[inline(always)]
unsafe fn copy_forward_misaligned_words(dest: *mut u8, src: *const u8, n: usize) {
    debug_assert!(n > 0 && n.is_multiple_of(WORD_SIZE));
    debug_assert!((src as usize) & WORD_MASK != 0);

    let mut dest_word = dest.cast::<Word>();
    let dest_end = dest.wrapping_add(n).cast::<Word>();
    let offset = src as usize & WORD_MASK;
    let shift = offset * 8;
    let mut src_aligned = src.wrapping_byte_sub(offset).cast::<Word>();
    let mut prev_word = load_aligned_end_partial(src_aligned, WORD_SIZE - offset);

    while dest_word.wrapping_add(1) < dest_end {
        src_aligned = src_aligned.wrapping_add(1);
        let cur_word = *src_aligned;
        let reassembled = if cfg!(target_endian = "little") {
            prev_word >> shift | cur_word << (WORD_SIZE * 8 - shift)
        } else {
            prev_word << shift | cur_word >> (WORD_SIZE * 8 - shift)
        };
        prev_word = cur_word;
        *dest_word = reassembled;
        dest_word = dest_word.wrapping_add(1);
    }

    src_aligned = src_aligned.wrapping_add(1);
    let cur_word = load_aligned_partial(src_aligned, offset);
    let reassembled = if cfg!(target_endian = "little") {
        prev_word >> shift | cur_word << (WORD_SIZE * 8 - shift)
    } else {
        prev_word << shift | cur_word >> (WORD_SIZE * 8 - shift)
    };
    *dest_word = reassembled;
}

#[inline(always)]
unsafe fn copy_forward_impl(mut dest: *mut u8, mut src: *const u8, mut n: usize) -> CopyPath {
    if n < WORD_COPY_THRESHOLD {
        copy_forward_bytes(dest, src, n);
        return CopyPath::ByteOnly;
    }

    let dest_misalignment = (dest as usize).wrapping_neg() & WORD_MASK;
    copy_forward_bytes(dest, src, dest_misalignment);
    dest = dest.wrapping_add(dest_misalignment);
    src = src.wrapping_add(dest_misalignment);
    n -= dest_misalignment;

    let n_words = n & !WORD_MASK;
    let path = if n_words == 0 {
        CopyPath::ByteOnly
    } else if (src as usize & WORD_MASK) == 0 {
        copy_forward_aligned_words(dest, src, n_words);
        CopyPath::AlignedWords
    } else {
        copy_forward_misaligned_words(dest, src, n_words);
        CopyPath::MisalignedWords
    };

    dest = dest.wrapping_add(n_words);
    src = src.wrapping_add(n_words);
    n -= n_words;
    copy_forward_bytes(dest, src, n);
    path
}

#[inline(always)]
unsafe fn copy_backward_bytes(mut dest: *mut u8, mut src: *const u8, n: usize) {
    let dest_start = dest.wrapping_sub(n);
    while dest_start < dest {
        dest = dest.wrapping_sub(1);
        src = src.wrapping_sub(1);
        *dest = *src;
    }
}

#[inline(always)]
unsafe fn copy_backward_aligned_words(dest: *mut u8, src: *const u8, n: usize) {
    let mut dest_word = dest.cast::<Word>();
    let mut src_word = src.cast::<Word>();
    let dest_start = dest.wrapping_sub(n).cast::<Word>();
    while dest_start < dest_word {
        dest_word = dest_word.wrapping_sub(1);
        src_word = src_word.wrapping_sub(1);
        *dest_word = *src_word;
    }
}

#[inline(always)]
unsafe fn copy_backward_misaligned_words(dest: *mut u8, src: *const u8, n: usize) {
    debug_assert!(n > 0 && n.is_multiple_of(WORD_SIZE));
    debug_assert!((src as usize) & WORD_MASK != 0);

    let mut dest_word = dest.cast::<Word>();
    let dest_start = dest.wrapping_sub(n).cast::<Word>();
    let offset = src as usize & WORD_MASK;
    let shift = offset * 8;
    let mut src_aligned = src.wrapping_byte_sub(offset).cast::<Word>();
    let mut prev_word = load_aligned_partial(src_aligned, offset);

    while dest_start.wrapping_add(1) < dest_word {
        src_aligned = src_aligned.wrapping_sub(1);
        let cur_word = *src_aligned;
        let reassembled = if cfg!(target_endian = "little") {
            prev_word << (WORD_SIZE * 8 - shift) | cur_word >> shift
        } else {
            prev_word >> (WORD_SIZE * 8 - shift) | cur_word << shift
        };
        prev_word = cur_word;
        dest_word = dest_word.wrapping_sub(1);
        *dest_word = reassembled;
    }

    src_aligned = src_aligned.wrapping_sub(1);
    let cur_word = load_aligned_end_partial(src_aligned, WORD_SIZE - offset);
    let reassembled = if cfg!(target_endian = "little") {
        prev_word << (WORD_SIZE * 8 - shift) | cur_word >> shift
    } else {
        prev_word >> (WORD_SIZE * 8 - shift) | cur_word << shift
    };
    dest_word = dest_word.wrapping_sub(1);
    *dest_word = reassembled;
}

#[inline(always)]
unsafe fn copy_backward_impl(dest: *mut u8, src: *const u8, mut n: usize) -> CopyPath {
    let mut dest = dest.wrapping_add(n);
    let mut src = src.wrapping_add(n);

    if n < WORD_COPY_THRESHOLD {
        copy_backward_bytes(dest, src, n);
        return CopyPath::ByteOnly;
    }

    let dest_misalignment = dest as usize & WORD_MASK;
    copy_backward_bytes(dest, src, dest_misalignment);
    dest = dest.wrapping_sub(dest_misalignment);
    src = src.wrapping_sub(dest_misalignment);
    n -= dest_misalignment;

    let n_words = n & !WORD_MASK;
    let path = if n_words == 0 {
        CopyPath::ByteOnly
    } else if (src as usize & WORD_MASK) == 0 {
        copy_backward_aligned_words(dest, src, n_words);
        CopyPath::AlignedWords
    } else {
        copy_backward_misaligned_words(dest, src, n_words);
        CopyPath::MisalignedWords
    };

    dest = dest.wrapping_sub(n_words);
    src = src.wrapping_sub(n_words);
    n -= n_words;
    copy_backward_bytes(dest, src, n);
    path
}

#[inline(always)]
unsafe fn memmove_impl(dest: *mut u8, src: *const u8, n: usize) -> (bool, CopyPath) {
    let delta = (dest as usize).wrapping_sub(src as usize);
    if delta >= n {
        (false, copy_forward_impl(dest, src, n))
    } else {
        (true, copy_backward_impl(dest, src, n))
    }
}

#[inline(always)]
unsafe fn set_bytes_bytes(mut s: *mut u8, c: u8, n: usize) {
    let end = s.wrapping_add(n);
    while s < end {
        *s = c;
        s = s.wrapping_add(1);
    }
}

#[inline(always)]
unsafe fn set_bytes_words(s: *mut u8, c: u8, n: usize) {
    let broadcast = u64::from_ne_bytes([c; WORD_SIZE]);
    let mut s_word = s.cast::<Word>();
    let end = s.wrapping_add(n).cast::<Word>();
    while s_word < end {
        *s_word = broadcast;
        s_word = s_word.wrapping_add(1);
    }
}

#[inline(always)]
unsafe fn set_bytes_impl(mut s: *mut u8, c: u8, mut n: usize) {
    if n >= WORD_COPY_THRESHOLD {
        let misalignment = (s as usize).wrapping_neg() & WORD_MASK;
        set_bytes_bytes(s, c, misalignment);
        s = s.wrapping_add(misalignment);
        n -= misalignment;

        let n_words = n & !WORD_MASK;
        set_bytes_words(s, c, n_words);
        s = s.wrapping_add(n_words);
        n -= n_words;
    }
    set_bytes_bytes(s, c, n);
}

#[inline(always)]
unsafe fn cmp_bytes(mut lhs: *const u8, mut rhs: *const u8, mut n: usize) -> i32 {
    while n > 0 {
        let lhs_byte = *lhs;
        let rhs_byte = *rhs;
        if lhs_byte != rhs_byte {
            return if lhs_byte < rhs_byte { -1 } else { 1 };
        }
        lhs = lhs.wrapping_add(1);
        rhs = rhs.wrapping_add(1);
        n -= 1;
    }
    0
}

#[inline(always)]
fn cmp_word(lhs: Word, rhs: Word) -> i32 {
    match Word::from_be(lhs).cmp(&Word::from_be(rhs)) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    }
}

#[inline(always)]
unsafe fn cmp_aligned_words(mut lhs: *const Word, mut rhs: *const Word, n: usize) -> i32 {
    let end = lhs.wrapping_byte_add(n);
    while lhs < end {
        let lhs_word = *lhs;
        let rhs_word = *rhs;
        if lhs_word != rhs_word {
            return cmp_word(lhs_word, rhs_word);
        }
        lhs = lhs.wrapping_add(1);
        rhs = rhs.wrapping_add(1);
    }
    0
}

#[inline(always)]
unsafe fn memcmp_impl(mut lhs: *const u8, mut rhs: *const u8, mut n: usize) -> i32 {
    let align_match = (((lhs as usize) ^ (rhs as usize)) & WORD_MASK) == 0;
    if align_match && n >= WORD_COMPARE_THRESHOLD {
        let prefix = (lhs as usize).wrapping_neg() & WORD_MASK;
        let prefix_cmp = cmp_bytes(lhs, rhs, prefix);
        if prefix_cmp != 0 {
            return prefix_cmp;
        }

        lhs = lhs.wrapping_add(prefix);
        rhs = rhs.wrapping_add(prefix);
        n -= prefix;

        let n_words = n & !WORD_MASK;
        if n_words != 0 {
            let word_cmp = cmp_aligned_words(lhs.cast::<Word>(), rhs.cast::<Word>(), n_words);
            if word_cmp != 0 {
                return word_cmp;
            }

            lhs = lhs.wrapping_add(n_words);
            rhs = rhs.wrapping_add(n_words);
            n -= n_words;
        }
    }

    cmp_bytes(lhs, rhs, n)
}

#[cfg(target_arch = "riscv64")]
#[no_mangle]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    let path = copy_forward_impl(dst, src, n);
    record_memcpy_call(dst, src, n, path);
    dst
}

#[cfg(target_arch = "riscv64")]
#[no_mangle]
pub unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    let (backward, path) = memmove_impl(dst, src, n);
    record_memmove_call(n, backward, path);
    dst
}

#[cfg(target_arch = "riscv64")]
#[no_mangle]
pub unsafe extern "C" fn memset(dst: *mut u8, val: i32, n: usize) -> *mut u8 {
    set_bytes_impl(dst, val as u8, n);
    dst
}

#[cfg(target_arch = "riscv64")]
#[no_mangle]
pub unsafe extern "C" fn memcmp(lhs: *const u8, rhs: *const u8, n: usize) -> i32 {
    memcmp_impl(lhs, rhs, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::{vec, vec::Vec};

    const LENGTHS: [usize; 18] = [
        0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 64, 65, 96,
    ];

    fn patterned(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (((i * 37) ^ (i >> 1) ^ 0x5a) & 0xff) as u8)
            .collect()
    }

    #[test]
    fn memcpy_impl_matches_copy_from_slice() {
        for src_off in 0..8 {
            for dst_off in 0..8 {
                for &len in &LENGTHS {
                    let src_storage = patterned(src_off + len + 16);
                    let mut actual = vec![0xa5; dst_off + len + 16];
                    let mut expected = actual.clone();
                    expected[dst_off..dst_off + len]
                        .copy_from_slice(&src_storage[src_off..src_off + len]);
                    unsafe {
                        copy_forward_impl(
                            actual.as_mut_ptr().add(dst_off),
                            src_storage.as_ptr().add(src_off),
                            len,
                        );
                    }
                    assert_eq!(
                        actual, expected,
                        "memcpy mismatch src_off={src_off} dst_off={dst_off} len={len}"
                    );
                }
            }
        }
    }

    #[test]
    fn memmove_impl_matches_copy_within() {
        let lengths = [0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64];
        for src_off in 0..24 {
            for dst_off in 0..24 {
                for &len in &lengths {
                    let total = 160;
                    if src_off + len > total || dst_off + len > total {
                        continue;
                    }
                    let mut actual = patterned(total);
                    let mut expected = actual.clone();
                    expected.copy_within(src_off..src_off + len, dst_off);
                    unsafe {
                        memmove_impl(
                            actual.as_mut_ptr().add(dst_off),
                            actual.as_ptr().add(src_off),
                            len,
                        );
                    }
                    assert_eq!(
                        actual, expected,
                        "memmove mismatch src_off={src_off} dst_off={dst_off} len={len}"
                    );
                }
            }
        }
    }

    #[test]
    fn memset_impl_matches_fill() {
        for off in 0..8 {
            for &len in &LENGTHS {
                let mut actual = vec![0x11; off + len + 16];
                let mut expected = actual.clone();
                expected[off..off + len].fill(0x7b);
                unsafe {
                    set_bytes_impl(actual.as_mut_ptr().add(off), 0x7b, len);
                }
                assert_eq!(actual, expected, "memset mismatch off={off} len={len}");
            }
        }
    }

    #[test]
    fn memcmp_impl_matches_slice_cmp() {
        for lhs_off in 0..8 {
            for rhs_off in 0..8 {
                for &len in &LENGTHS {
                    let lhs_storage = patterned(lhs_off + len + 16);
                    let mut rhs_storage = patterned(rhs_off + len + 16);
                    if len > 0 && ((lhs_off + rhs_off + len) & 1) == 0 {
                        rhs_storage[rhs_off + len / 2] ^= 1;
                    }
                    let lhs = &lhs_storage[lhs_off..lhs_off + len];
                    let rhs = &rhs_storage[rhs_off..rhs_off + len];
                    let expected = match lhs.cmp(rhs) {
                        core::cmp::Ordering::Less => -1,
                        core::cmp::Ordering::Equal => 0,
                        core::cmp::Ordering::Greater => 1,
                    };
                    let actual = unsafe {
                        memcmp_impl(
                            lhs_storage.as_ptr().add(lhs_off),
                            rhs_storage.as_ptr().add(rhs_off),
                            len,
                        )
                    };
                    assert_eq!(
                        actual, expected,
                        "memcmp mismatch lhs_off={lhs_off} rhs_off={rhs_off} len={len}"
                    );
                }
            }
        }
    }
}
