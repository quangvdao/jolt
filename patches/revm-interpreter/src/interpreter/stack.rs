use crate::InstructionResult;
use core::{fmt, ptr};
use primitives::U256;
use std::vec::Vec;

use super::StackTr;

#[inline(always)]
unsafe fn copy_u256_words(src: *const U256, dst: *mut U256) {
    let src_words = src.cast::<u64>();
    let dst_words = dst.cast::<u64>();
    // SAFETY: `U256` is repr(transparent) over `[u64; 4]`, and callers
    // ensure the source and destination do not overlap.
    ptr::write(dst_words.add(0), ptr::read(src_words.add(0)));
    ptr::write(dst_words.add(1), ptr::read(src_words.add(1)));
    ptr::write(dst_words.add(2), ptr::read(src_words.add(2)));
    ptr::write(dst_words.add(3), ptr::read(src_words.add(3)));
}

#[inline(always)]
unsafe fn swap_u256_words(lhs: *mut U256, rhs: *mut U256) {
    let lhs_words = lhs.cast::<u64>();
    let rhs_words = rhs.cast::<u64>();
    // SAFETY: `U256` is repr(transparent) over `[u64; 4]`, and callers
    // ensure the two pointers are in-bounds and non-overlapping.
    let lhs0 = ptr::read(lhs_words.add(0));
    let rhs0 = ptr::read(rhs_words.add(0));
    ptr::write(lhs_words.add(0), rhs0);
    ptr::write(rhs_words.add(0), lhs0);

    let lhs1 = ptr::read(lhs_words.add(1));
    let rhs1 = ptr::read(rhs_words.add(1));
    ptr::write(lhs_words.add(1), rhs1);
    ptr::write(rhs_words.add(1), lhs1);

    let lhs2 = ptr::read(lhs_words.add(2));
    let rhs2 = ptr::read(rhs_words.add(2));
    ptr::write(lhs_words.add(2), rhs2);
    ptr::write(rhs_words.add(2), lhs2);

    let lhs3 = ptr::read(lhs_words.add(3));
    let rhs3 = ptr::read(rhs_words.add(3));
    ptr::write(lhs_words.add(3), rhs3);
    ptr::write(rhs_words.add(3), lhs3);
}

/// EVM interpreter stack limit.
pub const STACK_LIMIT: usize = 1024;

/// EVM stack with [STACK_LIMIT] capacity of words.
#[derive(Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Stack {
    /// The underlying data of the stack.
    data: Vec<U256>,
}

impl fmt::Display for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, x) in self.data.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{x}")?;
        }
        f.write_str("]")
    }
}

impl Default for Stack {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Stack {
    fn clone(&self) -> Self {
        // Use `Self::new()` to ensure the cloned Stack is constructed with at least
        // STACK_LIMIT capacity, and then copy the data. This preserves the invariant
        // that Stack has sufficient capacity for operations that rely on it.
        let mut new_stack = Self::new();
        new_stack.data.extend_from_slice(&self.data);
        new_stack
    }
}

impl StackTr for Stack {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn data(&self) -> &[U256] {
        &self.data
    }

    #[inline]
    fn clear(&mut self) {
        self.data.clear();
    }

    #[inline]
    fn popn<const N: usize>(&mut self) -> Option<[U256; N]> {
        if self.len() < N {
            return None;
        }
        // SAFETY: Stack length is checked above.
        Some(unsafe { self.popn::<N>() })
    }

    #[inline]
    fn popn_top<const POPN: usize>(&mut self) -> Option<([U256; POPN], &mut U256)> {
        if self.len() < POPN + 1 {
            return None;
        }
        // SAFETY: Stack length is checked above.
        Some(unsafe { self.popn_top::<POPN>() })
    }

    #[inline]
    fn exchange(&mut self, n: usize, m: usize) -> bool {
        self.exchange(n, m)
    }

    #[inline]
    fn dup(&mut self, n: usize) -> bool {
        self.dup(n)
    }

    #[inline]
    fn push(&mut self, value: U256) -> bool {
        self.push(value)
    }

    #[inline]
    fn push_slice(&mut self, slice: &[u8]) -> bool {
        self.push_slice_(slice)
    }
}

impl Stack {
    /// Instantiate a new stack with the [default stack limit][STACK_LIMIT].
    #[inline]
    pub fn new() -> Self {
        Self {
            // SAFETY: Expansion functions assume that capacity is `STACK_LIMIT`.
            data: Vec::with_capacity(STACK_LIMIT),
        }
    }

    /// Instantiate a new invalid Stack.
    #[inline]
    pub fn invalid() -> Self {
        Self { data: Vec::new() }
    }

    /// Returns the length of the stack in words.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the underlying data buffer.
    #[inline]
    pub fn data(&self) -> &Vec<U256> {
        &self.data
    }

    /// Returns a mutable reference to the underlying data buffer.
    #[inline]
    pub fn data_mut(&mut self) -> &mut Vec<U256> {
        &mut self.data
    }

    /// Consumes the stack and returns the underlying data buffer.
    #[inline]
    pub fn into_data(self) -> Vec<U256> {
        self.data
    }

    /// Removes the topmost element from the stack and returns it, or `StackUnderflow` if it is
    /// empty.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn pop(&mut self) -> Result<U256, InstructionResult> {
        //Todo: can you likely instrincts to show the else branch is more likely to be hit
        let len = self.data.len();
        if primitives::hints_util::unlikely(len == 0) {
            Err(InstructionResult::StackUnderflow)
        } else {
            unsafe {
                self.data.set_len(len - 1);
                Ok(core::ptr::read(self.data.as_ptr().add(len - 1)))
            }
        }
    }

    /// Removes the topmost element from the stack and returns it.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn pop_unsafe(&mut self) -> U256 {
        assume!(!self.data.is_empty());
        self.data.pop().unwrap_unchecked()
    }

    /// Peeks the top of the stack.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn top_unsafe(&mut self) -> &mut U256 {
        assume!(!self.data.is_empty());
        self.data.last_mut().unwrap_unchecked()
    }

    /// Pops `N` values from the stack.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn popn<const N: usize>(&mut self) -> [U256; N] {
        assume!(self.data.len() >= N);
        core::array::from_fn(|_| unsafe { self.pop_unsafe() })
    }

    /// Pops `N` values from the stack and returns the top of the stack.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn popn_top<const POPN: usize>(&mut self) -> ([U256; POPN], &mut U256) {
        let result = self.popn::<POPN>();
        let top = self.top_unsafe();
        (result, top)
    }

    /// Push a new value onto the stack.
    ///
    /// If it will exceed the stack limit, returns false and leaves the stack
    /// unchanged.
    #[inline]
    #[must_use]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn push(&mut self, value: U256) -> bool {
        // In debug builds, verify we have sufficient capacity provisioned.
        debug_assert!(self.data.capacity() >= STACK_LIMIT);
        let len = self.data.len();
        if len == STACK_LIMIT {
            return false;
        }
        unsafe {
            let end = self.data.as_mut_ptr().add(len);
            core::ptr::write(end, value);
            self.data.set_len(len + 1);
        }
        true
    }

    /// Peek a value at given index for the stack, where the top of
    /// the stack is at index `0`. If the index is too large,
    /// `StackError::Underflow` is returned.
    #[inline]
    pub fn peek(&self, no_from_top: usize) -> Result<U256, InstructionResult> {
        if self.data.len() > no_from_top {
            Ok(self.data[self.data.len() - no_from_top - 1])
        } else {
            Err(InstructionResult::StackUnderflow)
        }
    }

    /// Duplicates the `N`th value from the top of the stack.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[inline]
    #[must_use]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn dup(&mut self, n: usize) -> bool {
        assume!(n > 0, "attempted to dup 0");
        let len = self.data.len();
        if len < n || len + 1 > STACK_LIMIT {
            false
        } else {
            // SAFETY: Check for out of bounds is done above and there is spare
            // capacity for the destination word. Copying four u64 limbs avoids
            // routing a 32-byte U256 move through a generic memcpy sequence.
            unsafe {
                let ptr = self.data.as_mut_ptr();
                copy_u256_words(ptr.add(len - n), ptr.add(len));
                self.data.set_len(len + 1);
            }
            true
        }
    }

    /// Swaps the topmost value with the `N`th value from the top.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn swap(&mut self, n: usize) -> bool {
        self.exchange(0, n)
    }

    /// Exchange two values on the stack.
    ///
    /// `n` is the first index, and the second index is calculated as `n + m`.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn exchange(&mut self, n: usize, m: usize) -> bool {
        assume!(m > 0, "overlapping exchange");
        let len = self.data.len();
        let n_m_index = n + m;
        if n_m_index >= len {
            return false;
        }
        // SAFETY: `n` and `n_m` are checked to be within bounds, and they
        // don't overlap. Swapping four u64 limbs avoids the generic
        // ptr::swap_nonoverlapping lowering that shows up as byte-wise memops
        // in Jolt traces.
        unsafe {
            let top = self.data.as_mut_ptr().add(len - 1);
            swap_u256_words(top.sub(n), top.sub(n_m_index));
        }
        true
    }

    /// Pushes an arbitrary length slice of bytes onto the stack, padding the last word with zeros
    /// if necessary.
    #[inline]
    pub fn push_slice(&mut self, slice: &[u8]) -> Result<(), InstructionResult> {
        if self.push_slice_(slice) {
            Ok(())
        } else {
            Err(InstructionResult::StackOverflow)
        }
    }

    /// Pushes an arbitrary length slice of bytes onto the stack, padding the last word with zeros
    /// if necessary.
    #[inline]
    fn push_slice_(&mut self, slice: &[u8]) -> bool {
        if slice.is_empty() {
            return true;
        }

        let n_words = slice.len().div_ceil(32);
        let new_len = self.data.len() + n_words;
        if new_len > STACK_LIMIT {
            return false;
        }

        // In debug builds, ensure underlying capacity is sufficient for the write.
        debug_assert!(self.data.capacity() >= new_len);

        // SAFETY: Length checked above.
        unsafe {
            let dst = self.data.as_mut_ptr().add(self.data.len()).cast::<u64>();
            self.data.set_len(new_len);

            let mut i = 0;

            // Write full words
            let words = slice.chunks_exact(32);
            let partial_last_word = words.remainder();
            for word in words {
                // Note: We unroll `U256::from_be_bytes` here to write directly into the buffer,
                // instead of creating a 32 byte array on the stack and then copying it over.
                for l in word.rchunks_exact(8) {
                    dst.add(i).write(u64::from_be_bytes(l.try_into().unwrap()));
                    i += 1;
                }
            }

            if partial_last_word.is_empty() {
                return true;
            }

            // Write limbs of partial last word
            let limbs = partial_last_word.rchunks_exact(8);
            let partial_last_limb = limbs.remainder();
            for l in limbs {
                dst.add(i).write(u64::from_be_bytes(l.try_into().unwrap()));
                i += 1;
            }

            // Write partial last limb by padding with zeros
            if !partial_last_limb.is_empty() {
                let mut tmp = [0u8; 8];
                tmp[8 - partial_last_limb.len()..].copy_from_slice(partial_last_limb);
                dst.add(i).write(u64::from_be_bytes(tmp));
                i += 1;
            }

            debug_assert_eq!(i.div_ceil(4), n_words, "wrote too much");

            // Zero out upper bytes of last word
            let m = i % 4; // 32 / 8
            if m != 0 {
                dst.add(i).write_bytes(0, 4 - m);
            }
        }

        true
    }

    /// Set a value at given index for the stack, where the top of the
    /// stack is at index `0`. If the index is too large,
    /// `StackError::Underflow` is returned.
    #[inline]
    pub fn set(&mut self, no_from_top: usize, val: U256) -> Result<(), InstructionResult> {
        if self.data.len() > no_from_top {
            let len = self.data.len();
            self.data[len - no_from_top - 1] = val;
            Ok(())
        } else {
            Err(InstructionResult::StackUnderflow)
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Stack {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct StackSerde {
            data: Vec<U256>,
        }

        let mut stack = StackSerde::deserialize(deserializer)?;
        if stack.data.len() > STACK_LIMIT {
            return Err(serde::de::Error::custom(std::format!(
                "stack size exceeds limit: {} > {}",
                stack.data.len(),
                STACK_LIMIT
            )));
        }
        stack.data.reserve(STACK_LIMIT - stack.data.len());
        Ok(Self { data: stack.data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl FnOnce(&mut Stack)) {
        let mut stack = Stack::new();
        // Fill capacity with non-zero values
        unsafe {
            stack.data.set_len(STACK_LIMIT);
            stack.data.fill(U256::MAX);
            stack.data.set_len(0);
        }
        f(&mut stack);
    }

    #[test]
    fn push_slices() {
        // No-op
        run(|stack| {
            stack.push_slice(b"").unwrap();
            assert!(stack.data.is_empty());
        });

        // One word
        run(|stack| {
            stack.push_slice(&[42]).unwrap();
            assert_eq!(stack.data, [U256::from(42)]);
        });

        let n = 0x1111_2222_3333_4444_5555_6666_7777_8888_u128;
        run(|stack| {
            stack.push_slice(&n.to_be_bytes()).unwrap();
            assert_eq!(stack.data, [U256::from(n)]);
        });

        // More than one word
        run(|stack| {
            let b = [U256::from(n).to_be_bytes::<32>(); 2].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data, [U256::from(n); 2]);
        });

        run(|stack| {
            let b = [&[0; 32][..], &[42u8]].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data, [U256::ZERO, U256::from(42)]);
        });

        run(|stack| {
            let b = [&[0; 32][..], &n.to_be_bytes()].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data, [U256::ZERO, U256::from(n)]);
        });

        run(|stack| {
            let b = [&[0; 64][..], &n.to_be_bytes()].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data, [U256::ZERO, U256::ZERO, U256::from(n)]);
        });
    }

    #[test]
    fn stack_clone() {
        // Test cloning an empty stack
        let empty_stack = Stack::new();
        let cloned_empty = empty_stack.clone();
        assert_eq!(empty_stack, cloned_empty);
        assert_eq!(cloned_empty.len(), 0);
        assert_eq!(cloned_empty.data().capacity(), STACK_LIMIT);

        // Test cloning a partially filled stack
        let mut partial_stack = Stack::new();
        for i in 0..10 {
            assert!(partial_stack.push(U256::from(i)));
        }
        let mut cloned_partial = partial_stack.clone();
        assert_eq!(partial_stack, cloned_partial);
        assert_eq!(cloned_partial.len(), 10);
        assert_eq!(cloned_partial.data().capacity(), STACK_LIMIT);

        // Test that modifying the clone doesn't affect the original
        assert!(cloned_partial.push(U256::from(100)));
        assert_ne!(partial_stack, cloned_partial);
        assert_eq!(partial_stack.len(), 10);
        assert_eq!(cloned_partial.len(), 11);

        // Test cloning a full stack
        let mut full_stack = Stack::new();
        for i in 0..STACK_LIMIT {
            assert!(full_stack.push(U256::from(i)));
        }
        let mut cloned_full = full_stack.clone();
        assert_eq!(full_stack, cloned_full);
        assert_eq!(cloned_full.len(), STACK_LIMIT);
        assert_eq!(cloned_full.data().capacity(), STACK_LIMIT);

        // Test push to the full original or cloned stack should return StackOverflow
        assert!(!full_stack.push(U256::from(100)));
        assert!(!cloned_full.push(U256::from(100)));
    }
}
