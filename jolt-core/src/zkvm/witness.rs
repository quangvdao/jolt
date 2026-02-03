#![allow(static_mut_refs)]

use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use std::io::{Read, Write};
use tracer::instruction::{Cycle, RAMAccess};

use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::program::ProgramPreprocessing;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    zkvm::ram::remap_address,
};

use super::instruction::{CircuitFlags, LookupQuery};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /*  Twist/Shout witnesses */
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// Packed bytecode commitment chunk polynomial (lane chunk i).
    /// This is used by BytecodeClaimReduction; commitment + batching integration is staged separately.
    BytecodeChunk(usize),
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    RamRa(usize),
    /// Dory dense matrix polynomial for recursion (after prefix packing).
    DoryDenseMatrix,
    /// Trusted advice polynomial - committed before proving, verifier has commitment.
    /// Length cannot exceed max_trace_length.
    TrustedAdvice,
    /// Untrusted advice polynomial - committed during proving, commitment in proof.
    /// Length cannot exceed max_trace_length.
    UntrustedAdvice,
    /// Program image words polynomial (initial RAM image), committed in preprocessing for
    /// `ProgramMode::Committed` and opened via `ProgramImageClaimReduction`.
    ///
    /// This polynomial is NOT streamed from the execution trace (it is provided as an "extra"
    /// polynomial to the Stage 8 streaming RLC builder, similar to advice polynomials).
    ProgramImageInit,
}

/// Returns a list of symbols representing all committed polynomials.
pub fn all_committed_polynomials(one_hot_params: &OneHotParams) -> Vec<CommittedPolynomial> {
    let mut polynomials = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::InstructionRa(i));
    }
    for i in 0..one_hot_params.ram_d {
        polynomials.push(CommittedPolynomial::RamRa(i));
    }
    for i in 0..one_hot_params.bytecode_d {
        polynomials.push(CommittedPolynomial::BytecodeRa(i));
    }
    polynomials
}

impl CommittedPolynomial {
    /// Generate witness data and compute tier 1 commitment for a single row
    pub fn stream_witness_and_commit_rows<F, PCS>(
        &self,
        setup: &PCS::ProverSetup,
        preprocessing: &JoltSharedPreprocessing,
        program: &ProgramPreprocessing,
        row_cycles: &[Cycle],
        one_hot_params: &OneHotParams,
    ) -> <PCS as StreamingCommitmentScheme>::ChunkState
    where
        F: JoltField,
        PCS: StreamingCommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                PCS::process_chunk(setup, &row)
            }
            CommittedPolynomial::RamInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| match cycle.ram_access() {
                        RAMAccess::Write(write) => {
                            write.post_value as i128 - write.pre_value as i128
                        }
                        _ => 0,
                    })
                    .collect();
                PCS::process_chunk(setup, &row)
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = program.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeChunk(_) => {
                panic!("Bytecode chunk polynomials are not stream-committed yet")
            }
            CommittedPolynomial::RamRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.memory_layout,
                        )
                        .map(|address| one_hot_params.ram_address_chunk(address, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::DoryDenseMatrix => {
                panic!("DoryDenseMatrix is not generated from witness data")
            }
            CommittedPolynomial::TrustedAdvice
            | CommittedPolynomial::UntrustedAdvice
            | CommittedPolynomial::ProgramImageInit => {
                panic!("Advice polynomials should not use streaming witness generation")
            }
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(
        &self,
        program: &ProgramPreprocessing,
        memory_layout: &MemoryLayout,
        trace: &[Cycle],
        one_hot_params: Option<&OneHotParams>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = program.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::BytecodeChunk(_) => {
                panic!("Bytecode chunk polynomials are not supported by generate_witness yet")
            }
            CommittedPolynomial::RamRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| one_hot_params.ram_address_chunk(address, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RdInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RamInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::InstructionRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::DoryDenseMatrix => {
                panic!("DoryDenseMatrix is not generated from witness data")
            }
            CommittedPolynomial::TrustedAdvice
            | CommittedPolynomial::UntrustedAdvice
            | CommittedPolynomial::ProgramImageInit => {
                panic!("Advice polynomials should not use generate_witness")
            }
        }
    }

    pub fn get_onehot_k(&self, one_hot_params: &OneHotParams) -> Option<usize> {
        match self {
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => Some(one_hot_params.k_chunk),
            _ => None,
        }
    }
}

// =============================================================================
// Recursion polynomial hierarchy - dependently typed
// Ordering: G1, G2, GT; within each: add/mul first, then scalar mul/exp
// =============================================================================

/// Trait for term enums - enables generic iteration and indexing
pub trait TermEnum: Copy + Clone + Sized + 'static {
    const COUNT: usize;

    fn from_index(i: usize) -> Option<Self>;
    fn to_index(self) -> usize;
    fn name(self) -> &'static str;
}

// --- G1 operations ---

/// G1 addition: P + Q = R with edge cases
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum G1AddTerm {
    XP,
    YP,
    PIndicator,
    XQ,
    YQ,
    QIndicator,
    XR,
    YR,
    RIndicator,
    Lambda,
    InvDeltaX,
    IsDouble,
    IsInverse,
}

impl TermEnum for G1AddTerm {
    const COUNT: usize = 13;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::XP),
            1 => Some(Self::YP),
            2 => Some(Self::PIndicator),
            3 => Some(Self::XQ),
            4 => Some(Self::YQ),
            5 => Some(Self::QIndicator),
            6 => Some(Self::XR),
            7 => Some(Self::YR),
            8 => Some(Self::RIndicator),
            9 => Some(Self::Lambda),
            10 => Some(Self::InvDeltaX),
            11 => Some(Self::IsDouble),
            12 => Some(Self::IsInverse),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::XP => "x_p",
            Self::YP => "y_p",
            Self::PIndicator => "p_indicator",
            Self::XQ => "x_q",
            Self::YQ => "y_q",
            Self::QIndicator => "q_indicator",
            Self::XR => "x_r",
            Self::YR => "y_r",
            Self::RIndicator => "r_indicator",
            Self::Lambda => "lambda",
            Self::InvDeltaX => "inv_delta_x",
            Self::IsDouble => "is_double",
            Self::IsInverse => "is_inverse",
        }
    }
}

/// G1 scalar multiplication: double-and-add with infinity handling
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum G1ScalarMulTerm {
    XA,
    YA,
    XT,
    YT,
    XANext,
    YANext,
    TIndicator,
    AIndicator,
    Bit,
    XP,
    YP,
}

impl TermEnum for G1ScalarMulTerm {
    const COUNT: usize = 11;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::XA),
            1 => Some(Self::YA),
            2 => Some(Self::XT),
            3 => Some(Self::YT),
            4 => Some(Self::XANext),
            5 => Some(Self::YANext),
            6 => Some(Self::TIndicator),
            7 => Some(Self::AIndicator),
            8 => Some(Self::Bit),
            9 => Some(Self::XP),
            10 => Some(Self::YP),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::XA => "x_a",
            Self::YA => "y_a",
            Self::XT => "x_t",
            Self::YT => "y_t",
            Self::XANext => "x_a_next",
            Self::YANext => "y_a_next",
            Self::TIndicator => "t_indicator",
            Self::AIndicator => "a_indicator",
            Self::Bit => "bit",
            Self::XP => "x_p",
            Self::YP => "y_p",
        }
    }
}

// --- G2 operations ---

/// G2 addition: same as G1 but Fq2 coords split into c0/c1
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum G2AddTerm {
    XPC0,
    XPC1,
    YPC0,
    YPC1,
    PIndicator,
    XQC0,
    XQC1,
    YQC0,
    YQC1,
    QIndicator,
    XRC0,
    XRC1,
    YRC0,
    YRC1,
    RIndicator,
    LambdaC0,
    LambdaC1,
    InvDeltaXC0,
    InvDeltaXC1,
    IsDouble,
    IsInverse,
}

impl TermEnum for G2AddTerm {
    const COUNT: usize = 21;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::XPC0),
            1 => Some(Self::XPC1),
            2 => Some(Self::YPC0),
            3 => Some(Self::YPC1),
            4 => Some(Self::PIndicator),
            5 => Some(Self::XQC0),
            6 => Some(Self::XQC1),
            7 => Some(Self::YQC0),
            8 => Some(Self::YQC1),
            9 => Some(Self::QIndicator),
            10 => Some(Self::XRC0),
            11 => Some(Self::XRC1),
            12 => Some(Self::YRC0),
            13 => Some(Self::YRC1),
            14 => Some(Self::RIndicator),
            15 => Some(Self::LambdaC0),
            16 => Some(Self::LambdaC1),
            17 => Some(Self::InvDeltaXC0),
            18 => Some(Self::InvDeltaXC1),
            19 => Some(Self::IsDouble),
            20 => Some(Self::IsInverse),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::XPC0 => "x_p_c0",
            Self::XPC1 => "x_p_c1",
            Self::YPC0 => "y_p_c0",
            Self::YPC1 => "y_p_c1",
            Self::PIndicator => "p_indicator",
            Self::XQC0 => "x_q_c0",
            Self::XQC1 => "x_q_c1",
            Self::YQC0 => "y_q_c0",
            Self::YQC1 => "y_q_c1",
            Self::QIndicator => "q_indicator",
            Self::XRC0 => "x_r_c0",
            Self::XRC1 => "x_r_c1",
            Self::YRC0 => "y_r_c0",
            Self::YRC1 => "y_r_c1",
            Self::RIndicator => "r_indicator",
            Self::LambdaC0 => "lambda_c0",
            Self::LambdaC1 => "lambda_c1",
            Self::InvDeltaXC0 => "inv_delta_x_c0",
            Self::InvDeltaXC1 => "inv_delta_x_c1",
            Self::IsDouble => "is_double",
            Self::IsInverse => "is_inverse",
        }
    }
}

/// G2 scalar multiplication: same as G1 but Fq2 coords split into c0/c1
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum G2ScalarMulTerm {
    XAC0,
    XAC1,
    YAC0,
    YAC1,
    XTC0,
    XTC1,
    YTC0,
    YTC1,
    XANextC0,
    XANextC1,
    YANextC0,
    YANextC1,
    TIndicator,
    AIndicator,
    XPC0,
    XPC1,
    YPC0,
    YPC1,
    Bit,
}

impl TermEnum for G2ScalarMulTerm {
    const COUNT: usize = 19;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::XAC0),
            1 => Some(Self::XAC1),
            2 => Some(Self::YAC0),
            3 => Some(Self::YAC1),
            4 => Some(Self::XTC0),
            5 => Some(Self::XTC1),
            6 => Some(Self::YTC0),
            7 => Some(Self::YTC1),
            8 => Some(Self::XANextC0),
            9 => Some(Self::XANextC1),
            10 => Some(Self::YANextC0),
            11 => Some(Self::YANextC1),
            12 => Some(Self::TIndicator),
            13 => Some(Self::AIndicator),
            14 => Some(Self::XPC0),
            15 => Some(Self::XPC1),
            16 => Some(Self::YPC0),
            17 => Some(Self::YPC1),
            18 => Some(Self::Bit),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::XAC0 => "x_a_c0",
            Self::XAC1 => "x_a_c1",
            Self::YAC0 => "y_a_c0",
            Self::YAC1 => "y_a_c1",
            Self::XTC0 => "x_t_c0",
            Self::XTC1 => "x_t_c1",
            Self::YTC0 => "y_t_c0",
            Self::YTC1 => "y_t_c1",
            Self::XANextC0 => "x_a_next_c0",
            Self::XANextC1 => "x_a_next_c1",
            Self::YANextC0 => "y_a_next_c0",
            Self::YANextC1 => "y_a_next_c1",
            Self::TIndicator => "t_indicator",
            Self::AIndicator => "a_indicator",
            Self::XPC0 => "x_p_c0",
            Self::XPC1 => "x_p_c1",
            Self::YPC0 => "y_p_c0",
            Self::YPC1 => "y_p_c1",
            Self::Bit => "bit",
        }
    }
}

// --- GT operations ---

/// GT multiplication: a × b = c + Q·g
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum GtMulTerm {
    Lhs,
    Rhs,
    Result,
    Quotient,
}

impl TermEnum for GtMulTerm {
    const COUNT: usize = 4;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Lhs),
            1 => Some(Self::Rhs),
            2 => Some(Self::Result),
            3 => Some(Self::Quotient),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::Lhs => "lhs",
            Self::Rhs => "rhs",
            Self::Result => "result",
            Self::Quotient => "quotient",
        }
    }
}

/// GT exponentiation (packed, base-4): ρ_next = ρ^4 × base^digit + Q·g
/// Note: digit bits are public inputs. The base may be treated as a boundary constant or as a
/// committed row, depending on the recursion protocol variant.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum GtExpTerm {
    Rho,
    RhoNext,
    Quotient,
    Base,
    Base2,
    Base3,
    BaseSquareQuotient,
    BaseCubeQuotient,
}

impl TermEnum for GtExpTerm {
    const COUNT: usize = 8;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Rho),
            1 => Some(Self::RhoNext),
            2 => Some(Self::Quotient),
            3 => Some(Self::Base),
            4 => Some(Self::Base2),
            5 => Some(Self::Base3),
            6 => Some(Self::BaseSquareQuotient),
            7 => Some(Self::BaseCubeQuotient),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::Rho => "rho",
            Self::RhoNext => "rho_next",
            Self::Quotient => "quotient",
            Self::Base => "base",
            Self::Base2 => "base2",
            Self::Base3 => "base3",
            Self::BaseSquareQuotient => "base_square_quotient",
            Self::BaseCubeQuotient => "base_cube_quotient",
        }
    }
}

// --- Pairing operations (BN254) ---

/// Multi-Miller loop trace terms (packed 11-var MLE per pair).
///
/// These are scalar-valued columns used to prove the BN254 Miller loop computation inside
/// recursion (over the base field \( \mathbb{F}_q \)).
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum MultiMillerLoopTerm {
    F,
    FNext,
    Quotient,
    TXC0,
    TXC1,
    TYC0,
    TYC1,
    TXC0Next,
    TXC1Next,
    TYC0Next,
    TYC1Next,
    LambdaC0,
    LambdaC1,
    InvDeltaXC0,
    InvDeltaXC1,
    InvTwoYC0,
    InvTwoYC1,
    XP,
    YP,
    XQC0,
    XQC1,
    YQC0,
    YQC1,
    IsDouble,
    IsAdd,
    LVal,
}

impl TermEnum for MultiMillerLoopTerm {
    const COUNT: usize = 26;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::F),
            1 => Some(Self::FNext),
            2 => Some(Self::Quotient),
            3 => Some(Self::TXC0),
            4 => Some(Self::TXC1),
            5 => Some(Self::TYC0),
            6 => Some(Self::TYC1),
            7 => Some(Self::TXC0Next),
            8 => Some(Self::TXC1Next),
            9 => Some(Self::TYC0Next),
            10 => Some(Self::TYC1Next),
            11 => Some(Self::LambdaC0),
            12 => Some(Self::LambdaC1),
            13 => Some(Self::InvDeltaXC0),
            14 => Some(Self::InvDeltaXC1),
            15 => Some(Self::InvTwoYC0),
            16 => Some(Self::InvTwoYC1),
            17 => Some(Self::XP),
            18 => Some(Self::YP),
            19 => Some(Self::XQC0),
            20 => Some(Self::XQC1),
            21 => Some(Self::YQC0),
            22 => Some(Self::YQC1),
            23 => Some(Self::IsDouble),
            24 => Some(Self::IsAdd),
            25 => Some(Self::LVal),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::F => "f",
            Self::FNext => "f_next",
            Self::Quotient => "quotient",
            Self::TXC0 => "t_x_c0",
            Self::TXC1 => "t_x_c1",
            Self::TYC0 => "t_y_c0",
            Self::TYC1 => "t_y_c1",
            Self::TXC0Next => "t_x_c0_next",
            Self::TXC1Next => "t_x_c1_next",
            Self::TYC0Next => "t_y_c0_next",
            Self::TYC1Next => "t_y_c1_next",
            Self::LambdaC0 => "lambda_c0",
            Self::LambdaC1 => "lambda_c1",
            Self::InvDeltaXC0 => "inv_delta_x_c0",
            Self::InvDeltaXC1 => "inv_delta_x_c1",
            Self::InvTwoYC0 => "inv_two_y_c0",
            Self::InvTwoYC1 => "inv_two_y_c1",
            Self::XP => "x_p",
            Self::YP => "y_p",
            Self::XQC0 => "x_q_c0",
            Self::XQC1 => "x_q_c1",
            Self::YQC0 => "y_q_c0",
            Self::YQC1 => "y_q_c1",
            Self::IsDouble => "is_double",
            Self::IsAdd => "is_add",
            Self::LVal => "l_val",
        }
    }
}

// =============================================================================
// RecursionPoly - the main hierarchy enum
// =============================================================================

/// Dependently-typed recursion polynomial
/// Each operation variant carries its own term enum
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum RecursionPoly {
    // G1 operations
    /// G1 addition term polynomial across all constraints.
    ///
    /// Stage 2 appends a *single* opening per term at the point (r_c, r_x),
    /// rather than per-instance openings.
    G1Add { term: G1AddTerm },
    /// G1 scalar multiplication term polynomial across all constraints.
    ///
    /// A single term polynomial over a family-local constraint index `c`
    /// (plus any native step variables).
    G1ScalarMul { term: G1ScalarMulTerm },
    // G2 operations
    /// G2 addition term polynomial across all constraints.
    ///
    /// Stage 2 appends a *single* opening per term at the point,
    /// rather than per-instance openings.
    G2Add { term: G2AddTerm },
    /// G2 scalar multiplication term polynomial across all constraints.
    ///
    /// A single term polynomial over a family-local constraint index `c`
    /// plus the native 8-var scalar-mul step domain.
    G2ScalarMul { term: G2ScalarMulTerm },
    // GT operations
    /// GT multiplication term polynomial across all constraints.
    ///
    /// Stage 2 appends a *single* opening per term at the point (r_c, r_x),
    /// rather than per-instance openings.
    GtMul { term: GtMulTerm },
    /// Packed GT-exp term polynomial across all constraints.
    ///
    /// GT-exp witnesses are reduced to a shared Stage-2 point by `GtExpClaimReduction`,
    /// and wiring reads those reduced openings.
    GtExp { term: GtExpTerm },
    /// Multi-Miller loop term polynomial for a specific constraint instance (global constraint idx).
    ///
    /// This is used to prove BN254 pairing Miller loop computation inside recursion.
    MultiMillerLoop {
        term: MultiMillerLoopTerm,
        instance: usize,
    },
}

impl RecursionPoly {
    pub fn term_index(&self) -> usize {
        match self {
            Self::G1Add { term } => term.to_index(),
            Self::G1ScalarMul { term } => term.to_index(),
            Self::G2Add { term } => term.to_index(),
            Self::G2ScalarMul { term } => term.to_index(),
            Self::GtMul { term } => term.to_index(),
            Self::GtExp { term } => term.to_index(),
            Self::MultiMillerLoop { term, .. } => term.to_index(),
        }
    }
}

// =============================================================================
// Serialization (canonical + guest) for RecursionPoly and VirtualPolynomial
// =============================================================================

// RecursionPoly canonical encoding: (tag: u8, term_index: u32)
// Note: We use the "_FUSED" tag values for backward compatibility with serialized data.
const RECURSION_POLY_TAG_G1_ADD: u8 = 1; // was G1_ADD_FUSED
const RECURSION_POLY_TAG_G1_SCALAR_MUL: u8 = 12; // was G1_SCALAR_MUL_FUSED
const RECURSION_POLY_TAG_G2_ADD: u8 = 13; // was G2_ADD_FUSED
const RECURSION_POLY_TAG_G2_SCALAR_MUL: u8 = 14; // was G2_SCALAR_MUL_FUSED
const RECURSION_POLY_TAG_GT_MUL: u8 = 9; // was GT_MUL_FUSED
const RECURSION_POLY_TAG_GT_EXP: u8 = 10; // was GT_EXP_FUSED
const RECURSION_POLY_TAG_MULTI_MILLER_LOOP: u8 = 15;

impl CanonicalSerialize for RecursionPoly {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let (tag, term_index, instance) = match *self {
            RecursionPoly::G1Add { term } => (RECURSION_POLY_TAG_G1_ADD, term.to_index(), 0usize),
            RecursionPoly::G1ScalarMul { term } => {
                (RECURSION_POLY_TAG_G1_SCALAR_MUL, term.to_index(), 0usize)
            }
            RecursionPoly::G2Add { term } => (RECURSION_POLY_TAG_G2_ADD, term.to_index(), 0usize),
            RecursionPoly::G2ScalarMul { term } => {
                (RECURSION_POLY_TAG_G2_SCALAR_MUL, term.to_index(), 0usize)
            }
            RecursionPoly::GtMul { term } => (RECURSION_POLY_TAG_GT_MUL, term.to_index(), 0usize),
            RecursionPoly::GtExp { term } => (RECURSION_POLY_TAG_GT_EXP, term.to_index(), 0usize),
            RecursionPoly::MultiMillerLoop { term, instance } => (
                RECURSION_POLY_TAG_MULTI_MILLER_LOOP,
                term.to_index(),
                instance,
            ),
        };

        tag.serialize_with_mode(&mut writer, compress)?;
        (term_index as u32).serialize_with_mode(&mut writer, compress)?;
        (instance as u32).serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        // tag (u8) + term_index (u32) + instance (u32)
        1 + 4 + 4
    }
}

impl Valid for RecursionPoly {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for RecursionPoly {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let term_index = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let instance_u32 = u32::deserialize_with_mode(&mut reader, compress, validate)?;
        let instance = instance_u32 as usize;
        if instance as u32 != instance_u32 {
            return Err(SerializationError::InvalidData);
        }

        Ok(match tag {
            RECURSION_POLY_TAG_G1_ADD => Self::G1Add {
                term: G1AddTerm::from_index(term_index).ok_or(SerializationError::InvalidData)?,
            },
            RECURSION_POLY_TAG_G1_SCALAR_MUL => Self::G1ScalarMul {
                term: G1ScalarMulTerm::from_index(term_index)
                    .ok_or(SerializationError::InvalidData)?,
            },
            RECURSION_POLY_TAG_G2_ADD => Self::G2Add {
                term: G2AddTerm::from_index(term_index).ok_or(SerializationError::InvalidData)?,
            },
            RECURSION_POLY_TAG_G2_SCALAR_MUL => Self::G2ScalarMul {
                term: G2ScalarMulTerm::from_index(term_index)
                    .ok_or(SerializationError::InvalidData)?,
            },
            RECURSION_POLY_TAG_GT_MUL => Self::GtMul {
                term: GtMulTerm::from_index(term_index).ok_or(SerializationError::InvalidData)?,
            },
            RECURSION_POLY_TAG_GT_EXP => Self::GtExp {
                term: GtExpTerm::from_index(term_index).ok_or(SerializationError::InvalidData)?,
            },
            RECURSION_POLY_TAG_MULTI_MILLER_LOOP => Self::MultiMillerLoop {
                term: MultiMillerLoopTerm::from_index(term_index)
                    .ok_or(SerializationError::InvalidData)?,
                instance,
            },
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for RecursionPoly {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Match the canonical shape: tag + term_index + instance.
        let (tag, term_index, instance) = match *self {
            RecursionPoly::G1Add { term } => (RECURSION_POLY_TAG_G1_ADD, term.to_index(), 0usize),
            RecursionPoly::G1ScalarMul { term } => {
                (RECURSION_POLY_TAG_G1_SCALAR_MUL, term.to_index(), 0usize)
            }
            RecursionPoly::G2Add { term } => (RECURSION_POLY_TAG_G2_ADD, term.to_index(), 0usize),
            RecursionPoly::G2ScalarMul { term } => {
                (RECURSION_POLY_TAG_G2_SCALAR_MUL, term.to_index(), 0usize)
            }
            RecursionPoly::GtMul { term } => (RECURSION_POLY_TAG_GT_MUL, term.to_index(), 0usize),
            RecursionPoly::GtExp { term } => (RECURSION_POLY_TAG_GT_EXP, term.to_index(), 0usize),
            RecursionPoly::MultiMillerLoop { term, instance } => (
                RECURSION_POLY_TAG_MULTI_MILLER_LOOP,
                term.to_index(),
                instance,
            ),
        };
        tag.guest_serialize(w)?;
        let term_index_u32 = u32::try_from(term_index).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "RecursionPoly term_index overflow",
            )
        })?;
        term_index_u32.guest_serialize(w)?;
        let instance_u32 = u32::try_from(instance).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "RecursionPoly instance overflow",
            )
        })?;
        instance_u32.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for RecursionPoly {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let tag = u8::guest_deserialize(r)?;
        let term_index = u32::guest_deserialize(r)? as usize;
        let instance_u32 = u32::guest_deserialize(r)?;
        let instance = instance_u32 as usize;
        if instance as u32 != instance_u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "RecursionPoly instance overflow",
            ));
        }
        Ok(match tag {
            RECURSION_POLY_TAG_G1_ADD => Self::G1Add {
                term: G1AddTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid G1AddTerm index")
                })?,
            },
            RECURSION_POLY_TAG_G1_SCALAR_MUL => Self::G1ScalarMul {
                term: G1ScalarMulTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid G1ScalarMulTerm index",
                    )
                })?,
            },
            RECURSION_POLY_TAG_G2_ADD => Self::G2Add {
                term: G2AddTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid G2AddTerm index")
                })?,
            },
            RECURSION_POLY_TAG_G2_SCALAR_MUL => Self::G2ScalarMul {
                term: G2ScalarMulTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid G2ScalarMulTerm index",
                    )
                })?,
            },
            RECURSION_POLY_TAG_GT_MUL => Self::GtMul {
                term: GtMulTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid GtMulTerm index")
                })?,
            },
            RECURSION_POLY_TAG_GT_EXP => Self::GtExp {
                term: GtExpTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid GtExpTerm index")
                })?,
            },
            RECURSION_POLY_TAG_MULTI_MILLER_LOOP => Self::MultiMillerLoop {
                term: MultiMillerLoopTerm::from_index(term_index).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid MultiMillerLoopTerm index",
                    )
                })?,
                instance,
            },
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid RecursionPoly tag",
                ))
            }
        })
    }
}

impl GuestSerialize for VirtualPolynomial {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Match the canonical tags used in `zkvm/proof_serialization.rs`.
        match *self {
            VirtualPolynomial::PC => 0u8.guest_serialize(w),
            VirtualPolynomial::UnexpandedPC => 1u8.guest_serialize(w),
            VirtualPolynomial::NextPC => 2u8.guest_serialize(w),
            VirtualPolynomial::NextUnexpandedPC => 3u8.guest_serialize(w),
            VirtualPolynomial::NextIsNoop => 4u8.guest_serialize(w),
            VirtualPolynomial::NextIsVirtual => 5u8.guest_serialize(w),
            VirtualPolynomial::NextIsFirstInSequence => 6u8.guest_serialize(w),
            VirtualPolynomial::LeftLookupOperand => 7u8.guest_serialize(w),
            VirtualPolynomial::RightLookupOperand => 8u8.guest_serialize(w),
            VirtualPolynomial::LeftInstructionInput => 9u8.guest_serialize(w),
            VirtualPolynomial::RightInstructionInput => 10u8.guest_serialize(w),
            VirtualPolynomial::Product => 11u8.guest_serialize(w),
            VirtualPolynomial::ShouldJump => 12u8.guest_serialize(w),
            VirtualPolynomial::ShouldBranch => 13u8.guest_serialize(w),
            VirtualPolynomial::WritePCtoRD => 14u8.guest_serialize(w),
            VirtualPolynomial::WriteLookupOutputToRD => 15u8.guest_serialize(w),
            VirtualPolynomial::Imm => 16u8.guest_serialize(w),
            VirtualPolynomial::Rs1Value => 17u8.guest_serialize(w),
            VirtualPolynomial::Rs2Value => 18u8.guest_serialize(w),
            VirtualPolynomial::RdWriteValue => 19u8.guest_serialize(w),
            VirtualPolynomial::Rs1Ra => 20u8.guest_serialize(w),
            VirtualPolynomial::Rs2Ra => 21u8.guest_serialize(w),
            VirtualPolynomial::RdWa => 22u8.guest_serialize(w),
            VirtualPolynomial::LookupOutput => 23u8.guest_serialize(w),
            VirtualPolynomial::InstructionRafFlag => 24u8.guest_serialize(w),
            VirtualPolynomial::InstructionRa(i) => {
                25u8.guest_serialize(w)?;
                let i_u32 = u32::try_from(i).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "VirtualPolynomial::InstructionRa index overflow",
                    )
                })?;
                i_u32.guest_serialize(w)
            }
            VirtualPolynomial::RegistersVal => 26u8.guest_serialize(w),
            VirtualPolynomial::RamAddress => 27u8.guest_serialize(w),
            VirtualPolynomial::RamRa => 28u8.guest_serialize(w),
            VirtualPolynomial::RamReadValue => 29u8.guest_serialize(w),
            VirtualPolynomial::RamWriteValue => 30u8.guest_serialize(w),
            VirtualPolynomial::RamVal => 31u8.guest_serialize(w),
            VirtualPolynomial::RamValInit => 32u8.guest_serialize(w),
            VirtualPolynomial::RamValFinal => 33u8.guest_serialize(w),
            VirtualPolynomial::RamHammingWeight => 34u8.guest_serialize(w),
            VirtualPolynomial::UnivariateSkip => 35u8.guest_serialize(w),
            VirtualPolynomial::OpFlags(flags) => {
                36u8.guest_serialize(w)?;
                (flags as u8).guest_serialize(w)
            }
            VirtualPolynomial::InstructionFlags(flags) => {
                37u8.guest_serialize(w)?;
                (flags as u8).guest_serialize(w)
            }
            VirtualPolynomial::LookupTableFlag(flag) => {
                38u8.guest_serialize(w)?;
                let b = u8::try_from(flag).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "VirtualPolynomial::LookupTableFlag overflow",
                    )
                })?;
                b.guest_serialize(w)
            }
            VirtualPolynomial::BytecodeValStage(stage) => {
                39u8.guest_serialize(w)?;
                let b = u8::try_from(stage).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "VirtualPolynomial::BytecodeValStage overflow",
                    )
                })?;
                b.guest_serialize(w)
            }
            VirtualPolynomial::BytecodeReadRafAddrClaim => 40u8.guest_serialize(w),
            VirtualPolynomial::BooleanityAddrClaim => 41u8.guest_serialize(w),
            VirtualPolynomial::BytecodeClaimReductionIntermediate => 42u8.guest_serialize(w),
            VirtualPolynomial::ProgramImageInitContributionRw => 43u8.guest_serialize(w),
            VirtualPolynomial::ProgramImageInitContributionRaf => 44u8.guest_serialize(w),
            VirtualPolynomial::Recursion(poly) => {
                45u8.guest_serialize(w)?;
                poly.guest_serialize(w)
            }
        }
    }
}

impl GuestDeserialize for VirtualPolynomial {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let tag = u8::guest_deserialize(r)?;
        Ok(match tag {
            0 => Self::PC,
            1 => Self::UnexpandedPC,
            2 => Self::NextPC,
            3 => Self::NextUnexpandedPC,
            4 => Self::NextIsNoop,
            5 => Self::NextIsVirtual,
            6 => Self::NextIsFirstInSequence,
            7 => Self::LeftLookupOperand,
            8 => Self::RightLookupOperand,
            9 => Self::LeftInstructionInput,
            10 => Self::RightInstructionInput,
            11 => Self::Product,
            12 => Self::ShouldJump,
            13 => Self::ShouldBranch,
            14 => Self::WritePCtoRD,
            15 => Self::WriteLookupOutputToRD,
            16 => Self::Imm,
            17 => Self::Rs1Value,
            18 => Self::Rs2Value,
            19 => Self::RdWriteValue,
            20 => Self::Rs1Ra,
            21 => Self::Rs2Ra,
            22 => Self::RdWa,
            23 => Self::LookupOutput,
            24 => Self::InstructionRafFlag,
            25 => {
                let i = u32::guest_deserialize(r)? as usize;
                Self::InstructionRa(i)
            }
            26 => Self::RegistersVal,
            27 => Self::RamAddress,
            28 => Self::RamRa,
            29 => Self::RamReadValue,
            30 => Self::RamWriteValue,
            31 => Self::RamVal,
            32 => Self::RamValInit,
            33 => Self::RamValFinal,
            34 => Self::RamHammingWeight,
            35 => Self::UnivariateSkip,
            36 => {
                let d = u8::guest_deserialize(r)?;
                let flags = CircuitFlags::from_repr(d).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid CircuitFlags")
                })?;
                Self::OpFlags(flags)
            }
            37 => {
                let d = u8::guest_deserialize(r)?;
                let flags = InstructionFlags::from_repr(d).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid InstructionFlags")
                })?;
                Self::InstructionFlags(flags)
            }
            38 => {
                let d = u8::guest_deserialize(r)? as usize;
                Self::LookupTableFlag(d)
            }
            39 => {
                let d = u8::guest_deserialize(r)? as usize;
                Self::BytecodeValStage(d)
            }
            40 => Self::BytecodeReadRafAddrClaim,
            41 => Self::BooleanityAddrClaim,
            42 => Self::BytecodeClaimReductionIntermediate,
            43 => Self::ProgramImageInitContributionRw,
            44 => Self::ProgramImageInitContributionRaf,
            45 => Self::Recursion(RecursionPoly::guest_deserialize(r)?),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid VirtualPolynomial tag",
                ))
            }
        })
    }
}

// =============================================================================
// VirtualPolynomial
// =============================================================================

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    WritePCtoRD,
    WriteLookupOutputToRD,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRafFlag,
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags(CircuitFlags),
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
    // Program-image commitment variants
    BytecodeValStage(usize),
    BytecodeReadRafAddrClaim,
    BooleanityAddrClaim,
    BytecodeClaimReductionIntermediate,
    ProgramImageInitContributionRw,
    ProgramImageInitContributionRaf,
    // Recursion protocol virtual polynomials - hierarchical structure
    Recursion(RecursionPoly),
}

impl VirtualPolynomial {
    /// Construct a Multi-Miller loop virtual polynomial (identified by global `constraint_idx`).
    #[inline]
    pub fn multi_miller_loop(term: MultiMillerLoopTerm, constraint_idx: usize) -> Self {
        VirtualPolynomial::Recursion(RecursionPoly::MultiMillerLoop {
            term,
            instance: constraint_idx,
        })
    }

    #[inline]
    pub fn multi_miller_loop_f(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::F, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_f_next(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::FNext, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_x_c0(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TXC0, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_x_c0_next(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TXC0Next, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_x_c1(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TXC1, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_x_c1_next(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TXC1Next, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_y_c0(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TYC0, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_y_c0_next(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TYC0Next, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_y_c1(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TYC1, constraint_idx)
    }
    #[inline]
    pub fn multi_miller_loop_t_y_c1_next(constraint_idx: usize) -> Self {
        Self::multi_miller_loop(MultiMillerLoopTerm::TYC1Next, constraint_idx)
    }
}
