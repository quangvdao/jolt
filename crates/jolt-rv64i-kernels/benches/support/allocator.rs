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
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies the valid allocation layout required by System.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            Self::added(layout.size());
            Self::allocation();
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
