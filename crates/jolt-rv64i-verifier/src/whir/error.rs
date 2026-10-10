//! Typed faults shared by commitment verification and proving.

use crate::commitment::BitsGeometry;
use thiserror::Error;

/// The buffer or protocol component measured by a size or shape check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum WhirPart {
    #[error("rows")]
    Rows,
    #[error("column point")]
    ColumnPoint,
    #[error("cycle point")]
    CyclePoint,
    #[error("columns")]
    Columns,
    #[error("lane values")]
    LaneValues,
    #[error("levels")]
    Levels,
    #[error("rounds")]
    Rounds,
    #[error("final values")]
    FinalValues,
    #[error("leaves")]
    Leaves,
    #[error("digests")]
    Digests,
}

/// Failures are reported in protocol order: geometry, shapes, then each
/// level's counts, authentication and terminal equations.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum WhirError {
    #[error("unsupported trace exponent {log_T}")]
    UnsupportedGeometry { log_T: usize },
    #[error("geometry mismatch: expected {expected:?}, got {actual:?}")]
    GeometryMismatch {
        expected: BitsGeometry,
        actual: BitsGeometry,
    },
    #[error("{part} shape: expected {expected}, got {actual}")]
    Shape {
        part: WhirPart,
        expected: usize,
        actual: usize,
    },
    #[error("level {level} {part} count: expected {expected}, got {actual}")]
    CountMismatch {
        level: usize,
        part: WhirPart,
        expected: usize,
        actual: usize,
    },
    #[error("{part} length overflow")]
    LengthOverflow { part: WhirPart },
    #[error("{part} allocation failed")]
    Allocation { part: WhirPart },
    #[error("level {level} Merkle authentication failed")]
    MerkleAuthentication { level: usize },
    #[error("final code mismatch at position {position}")]
    FinalCodeMismatch { position: usize },
    #[error("commit sample mismatch")]
    CommitSampleMismatch,
    #[error("closing identity mismatch")]
    ClosingIdentityMismatch,
}

/// Computes a product of lengths without wrapping; the empty product is one.
pub fn checked_product(part: WhirPart, lengths: &[usize]) -> Result<usize, WhirError> {
    lengths.iter().try_fold(1usize, |product, length| {
        product
            .checked_mul(*length)
            .ok_or(WhirError::LengthOverflow { part })
    })
}

/// Returns an empty vector with capacity for `len` elements. The byte length
/// is checked before reservation, and reservation failure is recoverable.
pub fn try_vec<T>(part: WhirPart, len: usize) -> Result<Vec<T>, WhirError> {
    let _bytes = checked_product(part, &[len, std::mem::size_of::<T>()])?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| WhirError::Allocation { part })?;
    Ok(out)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert successful reservations")]
mod tests {
    use super::{checked_product, try_vec, WhirError, WhirPart};
    use std::error::Error;

    #[test]
    fn length_overflow_preserves_part() {
        assert_eq!(
            checked_product(WhirPart::Leaves, &[usize::MAX, 2]),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Leaves
            })
        );
        assert_eq!(
            try_vec::<u64>(WhirPart::Rows, usize::MAX),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Rows
            })
        );
    }

    #[test]
    fn allocation_failure_preserves_part() {
        assert_eq!(
            try_vec::<u8>(WhirPart::Digests, isize::MAX.unsigned_abs()),
            Err(WhirError::Allocation {
                part: WhirPart::Digests
            })
        );
    }

    #[test]
    fn size_helpers_contract() {
        assert_eq!(checked_product(WhirPart::Levels, &[]), Ok(1));
        assert_eq!(checked_product(WhirPart::Rows, &[32, 4, 8]), Ok(1024));
        let out = try_vec::<u64>(WhirPart::Rows, 16).unwrap();
        assert!(out.is_empty());
        assert!(out.capacity() >= 16);
        fn error_contract<T: Error + Send + Sync + 'static>() {}
        error_contract::<WhirError>();
        error_contract::<WhirPart>();
    }
}
