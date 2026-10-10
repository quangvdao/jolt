//! Ordered statement checks exclude unsupported dimensions, noncanonical memory,
//! excessive RAM or segments, misplaced image words, invalid final PCs and proof
//! shapes. The checked initial RAM contains only nonzero inputs and image words.

use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_program::preprocess::{compute_max_ram_k, PublicInitialRam, PublicIoMemory};
use jolt_rv64i_arith::{Layout, LayoutError};

use crate::{
    commitment::BitsCommitmentScheme, error::Rv64iVerifierError,
    preprocessing::VerifierPreprocessing, proof::Rv64iProof,
};

/// Maximum admitted trace exponent; checked before any trace-dependent dimension is used.
pub const LOG_T_MAX: u8 = 32;

#[derive(Clone, Debug, PartialEq)]
/// Public execution statement, admitted by `CheckedInputs` before transcript absorption.
/// It supplies the trace length, entry PC and public I/O; execution agreement is checked by the protocol relations.
pub struct Statement {
    /// The exponent of the number of cycles, admitted in `1..=LOG_T_MAX`.
    pub log_T: u8,
    /// The claimed PC of cycle zero, bound by the bytecode-address relation.
    pub entry_pc: u64,
    /// Canonical memory layout, public segments and panic flag; advice must be empty.
    pub device: JoltDevice,
}

/// Validated public geometry, I/O and canonical initial RAM, borrowing the statement and preprocessing.
/// `of_statement` establishes checks 1–7 of §7; `new` additionally checks the proof's canonical shape.
pub struct CheckedInputs<'a, S: BitsCommitmentScheme> {
    preprocessing: &'a VerifierPreprocessing<S>,
    statement: &'a Statement,
    layout: Layout,
    io: PublicIoMemory,
    initial_ram: Vec<(u64, u64)>,
    final_pc: u64,
}

impl<'a, S: BitsCommitmentScheme> CheckedInputs<'a, S> {
    /// Runs checks 1–7 of §7, returning the typed error of the first failed dimension, layout, segment, image or final-PC check.
    /// Only bounded chunk metadata allocates before dimensions, canonical memory, RAM bounds and segment lengths pass.
    pub fn of_statement(
        preprocessing: &'a VerifierPreprocessing<S>,
        statement: &'a Statement,
        log_K_ram: u8,
        final_pc: u64,
    ) -> Result<Self, Rv64iVerifierError> {
        if statement.log_T == 0 {
            return Err(Rv64iVerifierError::TraceExponentZero);
        }
        if statement.log_T > LOG_T_MAX {
            return Err(Rv64iVerifierError::TraceExponentTooLarge {
                log_T: statement.log_T,
            });
        }
        if log_K_ram < 5 {
            return Err(Rv64iVerifierError::RamExponentTooSmall { log_K_ram });
        }
        let bytecode = preprocessing.bytecode();
        let layout = Layout::new(
            bytecode.log_K(),
            usize::from(log_K_ram),
            bytecode.lowest_address(),
        )
        .map_err(|error| match error {
            LayoutError::RamRangeOverflow { .. } => Rv64iVerifierError::RamRangeOverflow,
            LayoutError::LogSizeOutOfRange { .. }
            | LayoutError::LowestAddressNotAligned { .. }
            | LayoutError::BitsRowOverflow { .. }
            | LayoutError::IndexOutOfRange { .. } => Rv64iVerifierError::Layout(error),
        })?;
        let device = &statement.device;
        if !device.trusted_advice.is_empty() || !device.untrusted_advice.is_empty() {
            return Err(Rv64iVerifierError::AdviceNotEmpty);
        }
        let m = &device.memory_layout;
        let canonical = MemoryLayout::try_new(&MemoryConfig {
            max_input_size: m.max_input_size,
            max_trusted_advice_size: m.max_trusted_advice_size,
            max_untrusted_advice_size: m.max_untrusted_advice_size,
            max_output_size: m.max_output_size,
            stack_size: m.stack_size,
            heap_size: m.heap_size,
            program_size: Some(m.program_size),
        })
        .map_err(Rv64iVerifierError::MemoryLayout)?;
        if canonical != *m {
            return Err(Rv64iVerifierError::NonCanonicalMemoryLayout);
        }
        if bytecode.lowest_address() != m.get_lowest_address() {
            return Err(Rv64iVerifierError::LowestAddressMismatch);
        }
        let max_ram_K = compute_max_ram_k(m).map_err(Rv64iVerifierError::RamBound)?;
        if u32::from(log_K_ram) > max_ram_K.ilog2() {
            return Err(Rv64iVerifierError::RamTooLarge);
        }
        if u64::try_from(device.inputs.len()).map_err(|_| Rv64iVerifierError::InputsTooLong)?
            > m.max_input_size
        {
            return Err(Rv64iVerifierError::InputsTooLong);
        }
        if u64::try_from(device.outputs.len()).map_err(|_| Rv64iVerifierError::OutputsTooLong)?
            > m.max_output_size
        {
            return Err(Rv64iVerifierError::OutputsTooLong);
        }
        let io = PublicIoMemory::new(device).map_err(Rv64iVerifierError::PublicIo)?;
        let inputs =
            PublicInitialRam::inputs_only(device).map_err(Rv64iVerifierError::InitialRam)?;
        let ram_K = 1_u64 << log_K_ram;
        if io.io_mask_end > u128::from(ram_K) {
            return Err(Rv64iVerifierError::IoRangeTooLarge);
        }
        for &(index, _) in preprocessing.image() {
            if u128::from(index) < io.io_mask_end {
                return Err(Rv64iVerifierError::ImageInsideIo { index });
            }
            if index >= ram_K {
                return Err(Rv64iVerifierError::ImageOutsideRam { index });
            }
        }
        let _ = bytecode
            .final_pc_index(final_pc)
            .map_err(Rv64iVerifierError::FinalPc)?;
        let mut initial_ram = Vec::new();
        for segment in inputs.segments {
            let start = u64::try_from(segment.start_index)
                .map_err(|_| Rv64iVerifierError::IoRangeTooLarge)?;
            for (offset, value) in (0_u64..).zip(segment.words) {
                if value != 0 {
                    initial_ram.push((start + offset, value));
                }
            }
        }
        initial_ram.extend_from_slice(preprocessing.image());
        Ok(Self {
            preprocessing,
            statement,
            layout,
            io,
            initial_ram,
            final_pc,
        })
    }

    /// Runs the statement checks and then checks canonical round counts, round messages and 256 column values.
    /// Returns the first statement error or `ProofShape` before any sum-check runs.
    pub fn new(
        preprocessing: &'a VerifierPreprocessing<S>,
        statement: &'a Statement,
        proof: &Rv64iProof<S>,
    ) -> Result<Self, Rv64iVerifierError> {
        let checked =
            Self::of_statement(preprocessing, statement, proof.log_K_ram, proof.final_pc)?;
        proof
            .validate_shape(checked.log_T(), checked.log_K_bytecode())
            .map_err(Rv64iVerifierError::ProofShape)?;
        Ok(checked)
    }

    /// The admitted statement; its execution claims are still obligations of the protocol relations.
    pub fn statement(&self) -> &Statement {
        self.statement
    }

    /// The public bytecode, canonical image digest and scheme setup checked against this statement.
    pub fn preprocessing(&self) -> &VerifierPreprocessing<S> {
        self.preprocessing
    }

    /// The validated bit-column and RAM geometry shared by the family relations.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The public I/O, panic and termination words and their checked I/O mask.
    pub fn io(&self) -> &PublicIoMemory {
        &self.io
    }

    /// Canonical increasing nonzero input words followed by the program image; outputs and status words are absent.
    pub fn initial_ram(&self) -> &[(u64, u64)] {
        &self.initial_ram
    }

    /// The admitted cycle exponent in `1..=LOG_T_MAX`.
    pub fn log_T(&self) -> usize {
        usize::from(self.statement.log_T)
    }

    /// The padded bytecode exponent in `1..=24`.
    pub fn log_K_bytecode(&self) -> usize {
        self.layout.log_K_bytecode()
    }

    /// The admitted RAM-word exponent in `5..=61`, bounded by the statement's memory layout.
    pub fn log_K_ram(&self) -> usize {
        self.layout.log_K_ram()
    }

    /// The successor PC after the last cycle, checked to name a valid bytecode row.
    pub fn final_pc(&self) -> u64 {
        self.final_pc
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests inspect literal valid fixtures and their malformed variants"
)]
pub(crate) mod tests {
    use std::io::Error as IoError;

    use common::constants::RAM_START_ADDRESS;
    use jolt_field::{One, Zero, F128};
    use jolt_poly::CompressedPoly;
    use jolt_program::image::decode::decode_instruction;
    use jolt_riscv::RV64IMAC_JOLT;
    use jolt_rv64i_arith::Bytecode;
    use jolt_sumcheck::committed::CommittedSumcheckProof;
    use jolt_sumcheck::{ClearProof, CompressedSumcheckProof, SumcheckProof};
    use jolt_transcript::Transcript;

    use super::*;
    use crate::commitment::{BitsGeometry, BitsOpening, BitsWire};
    use crate::error::{PreprocessingError, ProofDecodeError};

    pub(crate) struct EmptyWire;

    impl BitsWire for EmptyWire {
        fn write(&self, _out: &mut Vec<u8>) {}

        fn read(bytes: &[u8], _geometry: BitsGeometry) -> Option<Self> {
            bytes.is_empty().then_some(Self)
        }
    }

    pub(crate) struct FixtureScheme;

    impl BitsCommitmentScheme for FixtureScheme {
        type VerifierSetup = ();
        type Commitment = EmptyWire;
        type VerifierState = ();
        type OpeningProof = EmptyWire;
        type Error = IoError;

        fn verify_commit<T: Transcript<Challenge = F128>>(
            _setup: &(),
            _geometry: BitsGeometry,
            _commitment: &EmptyWire,
            _transcript: &mut T,
        ) -> Result<(), IoError> {
            Ok(())
        }

        fn verify_opening<T: Transcript<Challenge = F128>>(
            _setup: &(),
            _state: (),
            _opening: &BitsOpening<'_>,
            _proof: &EmptyWire,
            _transcript: &mut T,
        ) -> Result<(), IoError> {
            Ok(())
        }
    }

    fn statement(config: MemoryConfig) -> Statement {
        Statement {
            log_T: 1,
            entry_pc: RAM_START_ADDRESS,
            device: JoltDevice::new(&config),
        }
    }

    fn small_statement() -> Statement {
        statement(MemoryConfig {
            max_input_size: 0,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_output_size: 0,
            stack_size: 8,
            heap_size: 8,
            program_size: Some(8),
        })
    }

    pub(crate) fn bytecode(lowest_address: u64) -> Bytecode {
        let layout = Layout::new(1, 5, lowest_address).unwrap();
        let source = decode_instruction(0x6f, RAM_START_ADDRESS, false, RV64IMAC_JOLT).unwrap();
        Bytecode::preprocess(&[source], &layout).unwrap()
    }

    fn preprocess(
        statement: &Statement,
        image: Vec<(u64, u64)>,
    ) -> VerifierPreprocessing<FixtureScheme> {
        VerifierPreprocessing::new(
            bytecode(statement.device.memory_layout.get_lowest_address()),
            image,
            (),
        )
        .unwrap()
    }

    fn fixture_proof() -> Rv64iProof<FixtureScheme> {
        let mut bytes = vec![0, 5];
        bytes.extend_from_slice(&RAM_START_ADDRESS.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        // At (t,b,a)=(1,1,5), the eight fixed-width sections contain respectively
        // 33,22,39,23,25,8,3,259 elements, including their wire values.
        for elements in [33, 22, 39, 23, 25, 8, 3, 259] {
            bytes.extend(std::iter::repeat_n(0, elements * 16));
        }
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        Rv64iProof::from_bytes(&bytes, 1, 1).unwrap()
    }

    fn check(
        preprocessing: &VerifierPreprocessing<FixtureScheme>,
        statement: &Statement,
        proof: &Rv64iProof<FixtureScheme>,
    ) -> Result<(), Rv64iVerifierError> {
        CheckedInputs::new(preprocessing, statement, proof).map(|_| ())
    }

    #[test]
    fn statement_rejects_unsupported_trace_and_ram_dimensions() {
        let mut statement = small_statement();
        let preprocessing = preprocess(&statement, vec![(2, 0x6f)]);
        let mut proof = fixture_proof();
        statement.log_T = 0;
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::TraceExponentZero)
        ));
        statement.log_T = 33;
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::TraceExponentTooLarge { log_T: 33 })
        ));
        statement.log_T = 1;
        proof.log_K_ram = 4;
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::RamExponentTooSmall { log_K_ram: 4 })
        ));
        proof.log_K_ram = 62;
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::Layout(
                LayoutError::LogSizeOutOfRange { .. }
            ))
        ));
        proof.log_K_ram = 6;
        let near_limit = VerifierPreprocessing::<FixtureScheme>::new(
            bytecode(u64::MAX - 255),
            vec![(2, 0x6f)],
            (),
        )
        .unwrap();
        assert!(matches!(
            check(&near_limit, &statement, &proof),
            Err(Rv64iVerifierError::RamRangeOverflow)
        ));
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::RamTooLarge)
        ));
    }

    #[test]
    fn statement_rejects_advice_and_noncanonical_memory_regions() {
        let honest = small_statement();
        let preprocessing = preprocess(&honest, vec![(2, 0x6f)]);
        let proof = fixture_proof();
        for trusted in [true, false] {
            let mut statement = honest.clone();
            if trusted {
                statement.device.trusted_advice.push(1);
            } else {
                statement.device.untrusted_advice.push(1);
            }
            assert!(matches!(
                check(&preprocessing, &statement, &proof),
                Err(Rv64iVerifierError::AdviceNotEmpty)
            ));
        }
        let mut empty_mask = honest.clone();
        empty_mask.device.memory_layout.input_start = RAM_START_ADDRESS;
        empty_mask.device.memory_layout.input_end = RAM_START_ADDRESS;
        empty_mask.device.memory_layout.output_start = RAM_START_ADDRESS - 8;
        empty_mask.device.memory_layout.output_end = RAM_START_ADDRESS - 8;
        let mut overlapping = statement(MemoryConfig {
            max_input_size: 16,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_output_size: 16,
            stack_size: 8,
            heap_size: 8,
            program_size: Some(8),
        });
        overlapping.device.memory_layout.output_start =
            overlapping.device.memory_layout.input_start;
        overlapping.device.memory_layout.output_end = overlapping.device.memory_layout.input_end;
        let mut unaligned = honest.clone();
        unaligned.device.memory_layout.input_start += 1;
        for statement in [empty_mask, overlapping, unaligned] {
            assert!(matches!(
                check(&preprocessing, &statement, &proof),
                Err(Rv64iVerifierError::NonCanonicalMemoryLayout)
            ));
        }
    }

    #[test]
    fn statement_rejects_base_segments_io_image_and_final_pc() {
        let honest = small_statement();
        let preprocessing = preprocess(&honest, vec![(2, 0x6f)]);
        let mut proof = fixture_proof();
        let wrong_base = VerifierPreprocessing::<FixtureScheme>::new(
            bytecode(RAM_START_ADDRESS - 24),
            vec![(2, 0x6f)],
            (),
        )
        .unwrap();
        assert!(matches!(
            check(&wrong_base, &honest, &proof),
            Err(Rv64iVerifierError::LowestAddressMismatch)
        ));
        let mut input = honest.clone();
        input.device.inputs.push(1);
        assert!(matches!(
            check(&preprocessing, &input, &proof),
            Err(Rv64iVerifierError::InputsTooLong)
        ));
        let mut output = honest.clone();
        output.device.outputs.push(1);
        assert!(matches!(
            check(&preprocessing, &output, &proof),
            Err(Rv64iVerifierError::OutputsTooLong)
        ));
        let large_io = statement(MemoryConfig {
            max_input_size: 512,
            ..MemoryConfig {
                max_input_size: 0,
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
                max_output_size: 0,
                stack_size: 8,
                heap_size: 8,
                program_size: Some(8),
            }
        });
        let large_preprocessing = preprocess(&large_io, vec![(128, 0x6f)]);
        assert!(matches!(
            check(&large_preprocessing, &large_io, &proof),
            Err(Rv64iVerifierError::IoRangeTooLarge)
        ));
        let image_inside_io = preprocess(&honest, vec![(1, 0x6f)]);
        assert!(matches!(
            check(&image_inside_io, &honest, &proof),
            Err(Rv64iVerifierError::ImageInsideIo { index: 1 })
        ));
        let image_outside_ram = preprocess(&honest, vec![(32, 0x6f)]);
        assert!(matches!(
            check(&image_outside_ram, &honest, &proof),
            Err(Rv64iVerifierError::ImageOutsideRam { index: 32 })
        ));
        for final_pc in [RAM_START_ADDRESS + 4, RAM_START_ADDRESS + 2] {
            proof.final_pc = final_pc;
            assert!(matches!(
                check(&preprocessing, &honest, &proof),
                Err(Rv64iVerifierError::FinalPc(_))
            ));
        }
    }

    #[test]
    fn statement_rejects_noncanonical_proof_shapes_before_sumcheck() {
        let statement = small_statement();
        let preprocessing = preprocess(&statement, vec![(2, 0x6f)]);
        for length in [255, 257] {
            let mut proof = fixture_proof();
            proof.stage6b.values.0.resize(length, F128::zero());
            assert!(matches!(
                check(&preprocessing, &statement, &proof),
                Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))
            ));
        }
        for coefficients in [
            vec![],
            vec![F128::one(); 4],
            vec![F128::one(), F128::zero()],
        ] {
            let mut proof = fixture_proof();
            let SumcheckProof::Clear(ClearProof::Compressed(rounds)) = &mut proof.stage1.rounds
            else {
                panic!("fixture must hold compressed rounds");
            };
            rounds.round_polynomials[0] = CompressedPoly::new(coefficients);
            assert!(matches!(
                check(&preprocessing, &statement, &proof),
                Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))
            ));
        }
        let mut proof = fixture_proof();
        proof.stage1.rounds = SumcheckProof::Clear(ClearProof::default());
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))
        ));
        proof.stage1.rounds = SumcheckProof::Committed(CommittedSumcheckProof::default());
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))
        ));
        proof.stage1.rounds =
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof::default()));
        assert!(matches!(
            check(&preprocessing, &statement, &proof),
            Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))
        ));
    }

    #[test]
    fn initial_ram_is_nonzero_inputs_then_image_with_advice_capacity() {
        let mut statement = statement(MemoryConfig {
            max_input_size: 16,
            max_trusted_advice_size: 16,
            max_untrusted_advice_size: 0,
            max_output_size: 8,
            stack_size: 8,
            heap_size: 8,
            program_size: Some(8),
        });
        statement.device.inputs = vec![7, 0, 0, 0, 0, 0, 0, 0, 0];
        statement.device.outputs = vec![9];
        let preprocessing = preprocess(&statement, vec![(8, 0x6f)]);
        let checked = CheckedInputs::new(&preprocessing, &statement, &fixture_proof()).unwrap();
        assert_eq!(checked.initial_ram(), [(2, 7), (8, 0x6f)]);
        assert_eq!(checked.io().io_mask_start, 2);
        assert_eq!(checked.io().io_mask_end, 8);
    }

    #[test]
    fn preprocessing_rejects_repeated_out_of_order_and_zero_image_words() {
        for (image, expected) in [
            (
                vec![(2, 1), (2, 3)],
                PreprocessingError::RepeatedImageWord { index: 2 },
            ),
            (
                vec![(3, 1), (2, 3)],
                PreprocessingError::ImageOutOfOrder {
                    previous: 3,
                    index: 2,
                },
            ),
            (vec![(2, 0)], PreprocessingError::ZeroImageWord { index: 2 }),
        ] {
            let result = VerifierPreprocessing::<FixtureScheme>::new(
                bytecode(RAM_START_ADDRESS - 16),
                image,
                (),
            );
            assert_eq!(result.err(), Some(expected));
        }
    }

    #[test]
    fn preprocessing_digest_matches_literal_jal_program() {
        let preprocessing = preprocess(&small_statement(), vec![(2, 0x6f)]);
        // Independently hashed 146 bytes: LE b=1 and R−16; the valid row's
        // columns 1,2^52,R,0,R+4,R,1,1,1; 53 zero padding bytes; LE 1,2,0x6f.
        assert_eq!(
            *preprocessing.digest(),
            [
                0xea, 0xa2, 0x28, 0xd1, 0x47, 0xdb, 0x8f, 0x16, 0x9d, 0x4c, 0xc3, 0x6a, 0xe5, 0xd5,
                0xa8, 0xad, 0xd7, 0x25, 0xce, 0xb9, 0xf7, 0xd8, 0x9c, 0x98, 0x49, 0x5f, 0xc1, 0x9e,
                0xb5, 0xc2, 0x2b, 0xb1,
            ]
        );
    }
}
