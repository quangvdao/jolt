//! Address-phase batch of public bytecode read checking.
pub mod bytecode_read;
pub mod verify;

use jolt_field::JoltField;
use jolt_verifier::stages::relations::SumcheckBatch;

pub use bytecode_read::{
    BytecodeReadAddress, BytecodeReadAddressChallenges, BytecodeReadAddressInputClaims,
    BytecodeReadAddressOutputClaims, BytecodeReadPoints,
};

#[derive(SumcheckBatch)]
pub struct Stage6aSumchecks<F: JoltField> {
    pub bytecode_read_address: BytecodeReadAddress<F>,
}
