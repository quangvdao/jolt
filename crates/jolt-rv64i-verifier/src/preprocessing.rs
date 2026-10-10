//! Public bytecode, canonical program image and their 32-byte preprocessing digest.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_rv64i_arith::{Bytecode, BytecodeColumn};

use crate::{commitment::BitsCommitmentScheme, error::PreprocessingError};

pub struct VerifierPreprocessing<S: BitsCommitmentScheme> {
    bytecode: Bytecode,
    image: Vec<(u64, u64)>,
    digest: [u8; 32],
    scheme: S::VerifierSetup,
}

impl<S: BitsCommitmentScheme> VerifierPreprocessing<S> {
    /// Admits only nonzero image words in strictly increasing index order.
    /// The digest encodes the bytecode exponent and base, every padded row in
    /// column order, then the image length and its index/value pairs.
    pub fn new(
        bytecode: Bytecode,
        image: Vec<(u64, u64)>,
        scheme: S::VerifierSetup,
    ) -> Result<Self, PreprocessingError> {
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

    pub fn bytecode(&self) -> &Bytecode {
        &self.bytecode
    }

    pub fn image(&self) -> &[(u64, u64)] {
        &self.image
    }

    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn scheme(&self) -> &S::VerifierSetup {
        &self.scheme
    }
}
