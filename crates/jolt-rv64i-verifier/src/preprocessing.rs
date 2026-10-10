//! Public bytecode, canonical program image and their 32-byte preprocessing digest.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_rv64i_arith::{Bytecode, BytecodeColumn};
use std::sync::Arc;

use crate::{commitment::BitsCommitmentScheme, error::PreprocessingError};

/// Shared public bytecode, a canonical nonzero image and the setup used by the bit-table scheme.
/// `new` validates the image and hashes the bytecode contents independently of its shared ownership.
pub struct VerifierPreprocessing<S: BitsCommitmentScheme> {
    bytecode: Arc<Bytecode>,
    image: Vec<(u64, u64)>,
    digest: [u8; 32],
    scheme: S::VerifierSetup,
}

impl<S: BitsCommitmentScheme> VerifierPreprocessing<S> {
    /// Retains shared bytecode and hashes its padded rows, base and canonical image.
    /// Returns an error for repeated or decreasing image indices, zero image words or unencodable lengths.
    pub fn new(
        bytecode: impl Into<Arc<Bytecode>>,
        image: Vec<(u64, u64)>,
        scheme: S::VerifierSetup,
    ) -> Result<Self, PreprocessingError> {
        let bytecode = bytecode.into();
        let mut previous = None;
        for &(index, value) in &image {
            if let Some(previous) = previous {
                if index == previous {
                    return Err(PreprocessingError::RepeatedImageWord { index });
                }
                if index < previous {
                    return Err(PreprocessingError::ImageOutOfOrder { previous, index });
                }
            }
            if value == 0 {
                return Err(PreprocessingError::ZeroImageWord { index });
            }
            previous = Some(index);
        }
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(
            u64::try_from(bytecode.log_K())
                .map_err(|_| PreprocessingError::Length)?
                .to_le_bytes(),
        );
        hasher.update(bytecode.lowest_address().to_le_bytes());
        for row in bytecode.rows() {
            for column in BytecodeColumn::ALL {
                let value = row.column(column);
                match column {
                    BytecodeColumn::Valid => hasher.update([u8::from(value != 0)]),
                    BytecodeColumn::Variant
                    | BytecodeColumn::PC
                    | BytecodeColumn::Imm
                    | BytecodeColumn::FallThroughPC
                    | BytecodeColumn::PCPlusImm => hasher.update(value.to_le_bytes()),
                    BytecodeColumn::Rs1Ra | BytecodeColumn::Rs2Ra | BytecodeColumn::RdWa => {
                        hasher.update(
                            u32::try_from(value)
                                .map_err(|_| PreprocessingError::Length)?
                                .to_le_bytes(),
                        );
                    }
                }
            }
        }
        hasher.update(
            u64::try_from(image.len())
                .map_err(|_| PreprocessingError::Length)?
                .to_le_bytes(),
        );
        for &(index, value) in &image {
            hasher.update(index.to_le_bytes());
            hasher.update(value.to_le_bytes());
        }
        Ok(Self {
            bytecode,
            image,
            digest: hasher.finalize().into(),
            scheme,
        })
    }

    /// The padded public bytecode whose contents are bound by `digest`.
    pub fn bytecode(&self) -> &Bytecode {
        &self.bytecode
    }

    /// The retained handle, which a host can clone to share the same table with its witness.
    pub fn shared_bytecode(&self) -> &Arc<Bytecode> {
        &self.bytecode
    }

    /// Nonzero program words in strictly increasing RAM-word index order.
    pub fn image(&self) -> &[(u64, u64)] {
        &self.image
    }

    /// The 32-byte preprocessing digest in the preamble, encoded as specified in §13.
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The setup borrowed by the scheme's commitment and opening checks.
    pub fn scheme(&self) -> &S::VerifierSetup {
        &self.scheme
    }
}
