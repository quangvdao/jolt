//! Allocation counters over requested bytes, relative to the measured phase's baseline.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct CountingAllocator;

pub struct AllocationAllowance {
    pub allocs: usize,
    pub bytes: usize,
}

// Rayon lazily boxes a 64-byte mutex and a 48-byte condition variable per worker.
pub const RAYON_WORKER_ALLOWANCE: AllocationAllowance = AllocationAllowance {
    allocs: 2,
    bytes: 64 + 48,
};

const RECORD_CAPACITY: usize = 4096;
static RECORD_THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
static RECORD_PHASE: AtomicUsize = AtomicUsize::new(0);
static RECORD_COUNT: AtomicUsize = AtomicUsize::new(0);
static RECORDS: [AtomicEntry; RECORD_CAPACITY] = [const { AtomicEntry::new() }; RECORD_CAPACITY];

struct AtomicEntry {
    size: AtomicUsize,
    phase: AtomicUsize,
}
impl AtomicEntry {
    const fn new() -> Self {
        Self {
            size: AtomicUsize::new(0),
            phase: AtomicUsize::new(0),
        }
    }
}

/// Lock-free inventory interface. Begin and read only while measured workers
/// are idle; stop after joining measured work, before reading the entries.
pub trait AllocationRecorder {
    fn begin(threshold: usize);
    fn phase(phase: usize);
    fn stop();
    fn overflow() -> usize;
    fn entries() -> impl Iterator<Item = (usize, usize)>;
    fn record(size: usize);
}

// The allocator is also included by private integration-test modules that use
// totals only. Keep the inventory entry points reachable in those consumers.
const _: fn(usize) = CountingAllocator::begin;
const _: fn(usize) = CountingAllocator::phase;
const _: fn() = CountingAllocator::stop;
const _: fn() -> usize = CountingAllocator::overflow;
const _: fn() = || {
    let _ = CountingAllocator::entries();
};

impl AllocationRecorder for CountingAllocator {
    fn begin(threshold: usize) {
        RECORD_COUNT.store(0, Ordering::Relaxed);
        RECORD_PHASE.store(0, Ordering::Relaxed);
        RECORD_THRESHOLD.store(threshold, Ordering::Relaxed);
    }

    fn phase(phase: usize) {
        RECORD_PHASE.store(phase, Ordering::Relaxed);
    }

    fn stop() {
        RECORD_THRESHOLD.store(usize::MAX, Ordering::Relaxed);
    }

    fn overflow() -> usize {
        RECORD_COUNT
            .load(Ordering::Relaxed)
            .saturating_sub(RECORD_CAPACITY)
    }

    fn entries() -> impl Iterator<Item = (usize, usize)> {
        let count = RECORD_COUNT.load(Ordering::Relaxed).min(RECORD_CAPACITY);
        RECORDS[..count].iter().map(|entry| {
            (
                entry.size.load(Ordering::Relaxed),
                entry.phase.load(Ordering::Relaxed),
            )
        })
    }

    fn record(size: usize) {
        if size < RECORD_THRESHOLD.load(Ordering::Relaxed) {
            return;
        }
        let index = RECORD_COUNT.fetch_add(1, Ordering::Relaxed);
        if let Some(entry) = RECORDS.get(index) {
            entry.size.store(size, Ordering::Relaxed);
            entry
                .phase
                .store(RECORD_PHASE.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }
}

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static MEASURING: AtomicBool = AtomicBool::new(false);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

impl CountingAllocator {
    /// Requested bytes currently live, including storage outside a measurement.
    pub fn live_bytes() -> usize {
        LIVE.load(Ordering::Relaxed)
    }

    fn added(bytes: usize) {
        let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if MEASURING.load(Ordering::Relaxed) {
            let _ = PEAK.fetch_max(live, Ordering::Relaxed);
        }
    }

    fn allocation() {
        if MEASURING.load(Ordering::Relaxed) {
            let _ = ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// SAFETY: All requests preserve System's layouts, pointers and ownership; counters
// use only atomics and never allocate or inspect the allocated memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies the valid allocation layout required by System.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            Self::added(layout.size());
            Self::allocation();
            Self::record(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies the valid allocation layout required by System.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            Self::added(layout.size());
            Self::allocation();
            Self::record(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout are forwarded unchanged from GlobalAlloc's caller.
        unsafe { System.dealloc(pointer, layout) };
        let _ = LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The live pointer, original layout and new size meet GlobalAlloc's contract.
        let result = unsafe { System.realloc(pointer, layout, new_size) };
        if !result.is_null() {
            if new_size >= layout.size() {
                Self::added(new_size - layout.size());
            } else {
                let _ = LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
            Self::allocation();
            Self::record(new_size);
        }
        result
    }
}

pub struct AllocationMeasurement {
    baseline: usize,
}

pub struct AllocationStats {
    pub peak_bytes: usize,
    pub final_bytes: usize,
    pub allocs: usize,
}

impl AllocationMeasurement {
    /// Excludes already resident sources and the warmed thread pool.
    pub fn begin() -> Self {
        let baseline = LIVE.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        ALLOCS.store(0, Ordering::Relaxed);
        MEASURING.store(true, Ordering::Relaxed);
        Self { baseline }
    }

    pub fn finish(self) -> AllocationStats {
        MEASURING.store(false, Ordering::Relaxed);
        AllocationStats {
            peak_bytes: PEAK.load(Ordering::Relaxed).saturating_sub(self.baseline),
            final_bytes: LIVE.load(Ordering::Relaxed).saturating_sub(self.baseline),
            allocs: ALLOCS.load(Ordering::Relaxed),
        }
    }
}

impl Drop for AllocationMeasurement {
    fn drop(&mut self) {
        MEASURING.store(false, Ordering::Relaxed);
    }
}
