//! Packed bits become weighted field sums through lookup lifts or bucket read-out.
//!
//! Lifts sum a word's set-bit weights without field multiplication. Bucket views
//! borrow position-major sets from a bounded scratch pool; sets for multiple
//! selectors are concatenated, with word slots then positions within each set.
//! The pool owns element-wise XOR tree merging. Scatter plans count destinations
//! once and emit one row/weight pair per cycle into disjoint destination ranges.
//! Cycle chunks use the deterministic geometry of [`crate::par::CycleChunks`].

pub mod bits;
pub mod buckets;
pub mod lift;
pub mod pool;
pub mod scatter;
