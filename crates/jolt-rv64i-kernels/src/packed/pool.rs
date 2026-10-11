//! Bounded scratch arrays for sequential chunk bodies and a disjoint XOR tree.
//! A returned array retains its contribution until `ScratchPool::merge` empties
//! the pool. A chunk must start no Rayon work while it holds a scratch guard.

use jolt_field::F128;
use rayon::prelude::*;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};
use thiserror::Error;

/// Rejected scratch dimensions or ownership transitions.
#[derive(Debug, Error)]
pub enum PoolError {
    /// Every array allowed by the pool's worker bound is currently lent.
    #[error("all scratch arrays are lent")]
    Exhausted,
    /// An array is still owned by a chunk when an exclusive pass is requested.
    #[error("cannot access scratch while {lent} arrays are lent")]
    MergeWhileLent { lent: usize },
    /// A merge or zero-fill owns the arrays exclusively.
    #[error("an exclusive scratch pass is in progress")]
    MergeInProgress,
    /// The requested array length cannot be represented or reserved.
    #[error("cannot reserve a scratch array of {len} field elements")]
    Length { len: usize },
    /// A panic while the free list was locked prevents further checked use.
    #[error("the scratch free-list lock is poisoned")]
    Poisoned,
}

/// Scratch of one fixed length, bounded by the Rayon worker count at construction.
/// Construct and use a pool inside the same Rayon pool. A chunk body that lends
/// an array must remain sequential until the guard is dropped: nested Rayon work
/// can otherwise hold multiple arrays on one worker and exhaust the bound.
pub struct ScratchPool {
    len: usize,
    capacity: usize,
    state: Mutex<PoolState>,
}

struct PoolState {
    free: Vec<Vec<F128>>,
    lent: usize,
    exclusive: bool,
}

impl ScratchPool {
    /// Checks that `len` field elements fit in a Rust allocation and captures
    /// `rayon::current_num_threads()` as the maximum pool-owned array count.
    /// Arrays are reserved and zero-filled only when first lent.
    pub fn new(len: usize) -> Result<Self, PoolError> {
        if len > isize::MAX as usize / size_of::<F128>() {
            return Err(PoolError::Length { len });
        }
        let capacity = rayon::current_num_threads();
        let mut free = Vec::new();
        free.try_reserve_exact(capacity)
            .map_err(|_| PoolError::Length { len })?;
        Ok(Self {
            len,
            capacity,
            state: Mutex::new(PoolState {
                free,
                lent: 0,
                exclusive: false,
            }),
        })
    }

    /// Lends a retained array, or creates a zero array below the worker bound.
    /// At the bound this returns `Exhausted` without allocating or waiting for
    /// an array. The state lock is acquired once, never per element; a new
    /// array is allocated and zero-filled after reserving the loan and unlocking.
    pub fn take(&self) -> Result<ScratchGuard<'_>, PoolError> {
        let array = {
            let mut state = self.lock()?;
            if state.exclusive {
                return Err(PoolError::MergeInProgress);
            }
            if state.lent == self.capacity {
                return Err(PoolError::Exhausted);
            }
            state.lent += 1;
            state.free.pop()
        };
        let array = array.unwrap_or_else(|| Self::zeroed(self.len));
        Ok(ScratchGuard { pool: self, array })
    }

    /// Returns the XOR of all contributions in a stride-doubling tree, parallel
    /// over disjoint ranges, and empties the pool. A subsequent loan starts at
    /// zero. A nonempty merge transfers an existing array without allocating;
    /// an empty merge reserves one zero result. Lent arrays and concurrent
    /// merges are reported before any contribution is consumed.
    pub fn merge(&self) -> Result<Vec<F128>, PoolError> {
        let mut owned = self.detach()?;
        let arrays = &mut owned.arrays;
        let result = if arrays.is_empty() {
            Ok(Self::zeroed(self.len))
        } else {
            let mut stride = 1;
            while stride < arrays.len() {
                arrays
                    .par_chunks_mut(stride.saturating_mul(2))
                    .for_each(|pair| {
                        if pair.len() > stride {
                            let (left, right) = pair.split_at_mut(stride);
                            left[0]
                                .par_chunks_mut(4096)
                                .zip(right[0].par_chunks(4096))
                                .for_each(|(left, right)| {
                                    for (left, &right) in left.iter_mut().zip(right) {
                                        *left += right;
                                    }
                                });
                        }
                    });
                stride = stride.saturating_mul(2);
            }
            Ok(arrays.swap_remove(0))
        };
        arrays.clear();
        result
    }

    /// Zero-fills every existing pool-owned array in parallel over disjoint
    /// ranges without allocating or discarding its storage. Lent arrays and
    /// concurrent exclusive passes are rejected before any element is changed.
    pub fn zero(&self) -> Result<(), PoolError> {
        let mut owned = self.detach()?;
        owned.arrays.par_iter_mut().for_each(|array| {
            array.par_chunks_mut(4096).for_each(|chunk| {
                chunk.fill(F128::from_raw(0));
            });
        });
        Ok(())
    }

    /// Counts live free and lent arrays, for checking the scratch bound.
    /// A merge or zero-fill owns its arrays exclusively and returns `MergeInProgress`.
    pub fn allocated_arrays(&self) -> Result<usize, PoolError> {
        let state = self.lock()?;
        if state.exclusive {
            return Err(PoolError::MergeInProgress);
        }
        Ok(state.free.len() + state.lent)
    }

    fn detach(&self) -> Result<ScratchArrays<'_>, PoolError> {
        let mut state = self.lock()?;
        if state.exclusive {
            return Err(PoolError::MergeInProgress);
        }
        if state.lent != 0 {
            return Err(PoolError::MergeWhileLent { lent: state.lent });
        }
        state.exclusive = true;
        Ok(ScratchArrays {
            pool: self,
            arrays: std::mem::take(&mut state.free),
        })
    }

    fn restore(&self, arrays: Vec<Vec<F128>>) {
        // Destructors restore ownership even on unwind; checked operations
        // continue to report a poisoned free-list lock.
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        state.free = arrays;
        state.exclusive = false;
    }

    fn zeroed(len: usize) -> Vec<F128> {
        vec![F128::from_raw(0); len]
    }

    fn lock(&self) -> Result<MutexGuard<'_, PoolState>, PoolError> {
        self.state.lock().map_err(|_| PoolError::Poisoned)
    }
}

struct ScratchArrays<'a> {
    pool: &'a ScratchPool,
    arrays: Vec<Vec<F128>>,
}

impl Drop for ScratchArrays<'_> {
    fn drop(&mut self) {
        self.pool.restore(std::mem::take(&mut self.arrays));
    }
}

/// An exclusive scratch loan that returns its array with its contents on drop.
/// Element accesses require indices below the length given to `ScratchPool::new`.
pub struct ScratchGuard<'a> {
    pool: &'a ScratchPool,
    array: Vec<F128>,
}

impl Deref for ScratchGuard<'_> {
    type Target = [F128];

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl DerefMut for ScratchGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        // A destructor cannot report poison. Returning ownership preserves the
        // array bound; later checked operations still return PoolError::Poisoned.
        let mut state = self
            .pool
            .state
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        state.free.push(std::mem::take(&mut self.array));
        state.lent -= 1;
    }
}
