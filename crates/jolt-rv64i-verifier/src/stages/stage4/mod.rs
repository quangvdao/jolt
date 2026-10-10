//! Concrete relations for binary-field RV64I stage4.
pub mod ram_output_check;
pub mod ram_read_checking;
pub mod registers_read_checking;
pub mod verify;

use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::batch_geometry;
use crate::statement::LOG_T_MAX;
use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_field::{JoltField, Zero, F128};
use jolt_program::preprocess::PublicIoMemory;
use jolt_rv64i_arith::Layout;
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use ram_output_check::RamOutputCheck;
pub use ram_read_checking::RamReadChecking;
pub use registers_read_checking::RegistersReadChecking;
use std::sync::Arc;
pub use verify::Output;

/// Batch 4 places RAM addresses before cycles and aligns the last five addresses with registers.
#[derive(SumcheckBatch)]
pub struct Stage4Sumchecks<F: JoltField> {
    pub registers_read_checking: RegistersReadChecking<F>,
    pub ram_read_checking: RamReadChecking<F>,
    pub ram_output_check: RamOutputCheck<F>,
}

impl<F: JoltField> Stage4Sumchecks<F> {
    /// Establishes six bit coordinates, bounded cycle coordinates, and at least five RAM addresses, all low variable first.
    /// Rejects malformed points or I/O geometry and draws the output-check address challenge before member coefficients.
    pub fn new<T: Transcript<Challenge = F>>(
        layout: &Layout,
        r_bit: Vec<F>,
        r_3: Vec<F>,
        io: impl Into<Arc<PublicIoMemory>>,
        transcript: &mut T,
    ) -> Result<Self, PointsError> {
        let registers_read_checking = RegistersReadChecking::new(r_bit.clone(), r_3.clone())?;
        let ram_read_checking = RamReadChecking::new(layout, r_bit.clone(), r_3)?;
        let tau = transcript.challenge_vector(layout.log_K_ram());
        let ram_output_check = RamOutputCheck::new(tau, r_bit, io)?;
        Ok(Self {
            registers_read_checking,
            ram_read_checking,
            ram_output_check,
        })
    }
}

impl Stage4Sumchecks<F128> {
    /// Constructs bounded geometry instances without a witness or statement-sized public data.
    /// Rejects unsupported trace widths before allocation; concrete constructors establish the remaining point bounds.
    pub fn for_geometry(log_T: usize, layout: &Layout) -> Result<Self, Rv64iVerifierError> {
        if !(1..=usize::from(LOG_T_MAX)).contains(&log_T) {
            return Err(verify::term_error(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            })
            .into());
        }
        let memory_layout = MemoryLayout::try_new(&MemoryConfig {
            max_input_size: 0,
            max_output_size: 0,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            stack_size: 0,
            heap_size: 0,
            program_size: Some(8),
        })
        .map_err(Rv64iVerifierError::MemoryLayout)?;
        let io = PublicIoMemory::new(&JoltDevice {
            memory_layout,
            ..JoltDevice::default()
        })
        .map_err(Rv64iVerifierError::PublicIo)?;
        Ok(Self {
            registers_read_checking: RegistersReadChecking::new(
                vec![F128::zero(); 6],
                vec![F128::zero(); log_T],
            )
            .map_err(verify::term_error)?,
            ram_read_checking: RamReadChecking::new(
                layout,
                vec![F128::zero(); 6],
                vec![F128::zero(); log_T],
            )
            .map_err(verify::term_error)?,
            ram_output_check: RamOutputCheck::new(
                vec![F128::zero(); layout.log_K_ram()],
                vec![F128::zero(); 6],
                io,
            )
            .map_err(verify::term_error)?,
        })
    }
}
stage4_sumchecks_members!(batch_geometry);
