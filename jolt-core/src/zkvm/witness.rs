#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::InstructionFlags;
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
        program: &crate::zkvm::program::ProgramPreprocessing,
        row_cycles: &[tracer::instruction::Cycle],
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
                        tracer::instruction::RAMAccess::Write(write) => {
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
        program: &crate::zkvm::program::ProgramPreprocessing,
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
                            tracer::instruction::RAMAccess::Write(write) => {
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
}

impl TermEnum for G1ScalarMulTerm {
    const COUNT: usize = 9;

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
    Bit,
}

impl TermEnum for G2ScalarMulTerm {
    const COUNT: usize = 15;

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
            14 => Some(Self::Bit),
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
/// Note: digit/base are public inputs, not committed
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum GtExpTerm {
    Rho,
    RhoNext,
    Quotient,
}

impl TermEnum for GtExpTerm {
    const COUNT: usize = 3;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Rho),
            1 => Some(Self::RhoNext),
            2 => Some(Self::Quotient),
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
        }
    }
}

/// Multi-Miller loop: f_next = f^2 * l + Q*g
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

/// Frobenius map: output = input^q
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum FrobeniusTerm {
    Input,
    Output,
    FrobConst,
}

impl TermEnum for FrobeniusTerm {
    const COUNT: usize = 3;

    fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Input),
            1 => Some(Self::Output),
            2 => Some(Self::FrobConst),
            _ => None,
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::FrobConst => "frob_const",
        }
    }
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
    G1Add {
        term: G1AddTerm,
        instance: usize,
    },
    /// Fused G1 addition term polynomial across all constraints.
    ///
    /// This is a transitional variant used by the recursion fused-virtual-polynomials port:
    /// Stage 2 may append a *single* opening per term at the fused point (r_c, r_x),
    /// rather than per-instance openings.
    G1AddFused {
        term: G1AddTerm,
    },
    G1ScalarMul {
        term: G1ScalarMulTerm,
        instance: usize,
    },
    // G2 operations
    G2Add {
        term: G2AddTerm,
        instance: usize,
    },
    G2ScalarMul {
        term: G2ScalarMulTerm,
        instance: usize,
    },
    // GT operations
    GtMul {
        term: GtMulTerm,
        instance: usize,
    },
    GtExp {
        term: GtExpTerm,
        instance: usize,
    },
    MultiMillerLoop {
        term: MultiMillerLoopTerm,
        instance: usize,
    },
    Frobenius {
        term: FrobeniusTerm,
        instance: usize,
    },
}

impl RecursionPoly {
    pub fn instance(&self) -> usize {
        match self {
            Self::G1Add { instance, .. } => *instance,
            Self::G1AddFused { .. } => 0,
            Self::G1ScalarMul { instance, .. } => *instance,
            Self::G2Add { instance, .. } => *instance,
            Self::G2ScalarMul { instance, .. } => *instance,
            Self::GtMul { instance, .. } => *instance,
            Self::GtExp { instance, .. } => *instance,
            Self::MultiMillerLoop { instance, .. } => *instance,
            Self::Frobenius { instance, .. } => *instance,
        }
    }

    pub fn term_index(&self) -> usize {
        match self {
            Self::G1Add { term, .. } => term.to_index(),
            Self::G1AddFused { term } => term.to_index(),
            Self::G1ScalarMul { term, .. } => term.to_index(),
            Self::G2Add { term, .. } => term.to_index(),
            Self::G2ScalarMul { term, .. } => term.to_index(),
            Self::GtMul { term, .. } => term.to_index(),
            Self::GtExp { term, .. } => term.to_index(),
            Self::MultiMillerLoop { term, .. } => term.to_index(),
            Self::Frobenius { term, .. } => term.to_index(),
        }
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
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
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
    /// Staged scalar program-image contribution at `r_address_rw` (Stage 4).
    ProgramImageInitContributionRw,
    /// Staged scalar program-image contribution at `r_address_raf` (Stage 4), when the two
    /// address points differ.
    ProgramImageInitContributionRaf,
    // Recursion protocol virtual polynomials - hierarchical structure
    Recursion(RecursionPoly),
    // Dory sparse constraint matrix - virtualized in Stage 2, dense version committed in Stage 3
    DorySparseConstraintMatrix,
}

// =============================================================================
// VirtualPolynomial convenience constructors for backward compatibility
// =============================================================================

impl VirtualPolynomial {
    // --- GT multiplication ---
    pub fn gt_mul_lhs(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Lhs,
            instance: i,
        })
    }
    pub fn gt_mul_rhs(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Rhs,
            instance: i,
        })
    }
    pub fn gt_mul_result(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Result,
            instance: i,
        })
    }
    pub fn gt_mul_quotient(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Quotient,
            instance: i,
        })
    }

    // --- GT exponentiation (packed) ---
    pub fn gt_exp_rho(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtExp {
            term: GtExpTerm::Rho,
            instance: i,
        })
    }
    pub fn gt_exp_rho_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtExp {
            term: GtExpTerm::RhoNext,
            instance: i,
        })
    }
    pub fn gt_exp_quotient(i: usize) -> Self {
        Self::Recursion(RecursionPoly::GtExp {
            term: GtExpTerm::Quotient,
            instance: i,
        })
    }

    // --- Multi-Miller loop ---
    pub fn multi_miller_loop_f(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::F,
            instance: i,
        })
    }
    pub fn multi_miller_loop_f_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::FNext,
            instance: i,
        })
    }
    pub fn multi_miller_loop_quotient(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::Quotient,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_x_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TXC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_x_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TXC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_y_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TYC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_y_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TYC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_x_c0_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TXC0Next,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_x_c1_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TXC1Next,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_y_c0_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TYC0Next,
            instance: i,
        })
    }
    pub fn multi_miller_loop_t_y_c1_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::TYC1Next,
            instance: i,
        })
    }
    pub fn multi_miller_loop_lambda_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::LambdaC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_lambda_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::LambdaC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_inv_delta_x_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::InvDeltaXC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_inv_delta_x_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::InvDeltaXC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_inv_two_y_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::InvTwoYC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_inv_two_y_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::InvTwoYC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_x_p(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::XP,
            instance: i,
        })
    }
    pub fn multi_miller_loop_y_p(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::YP,
            instance: i,
        })
    }
    pub fn multi_miller_loop_x_q_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::XQC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_x_q_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::XQC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_y_q_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::YQC0,
            instance: i,
        })
    }
    pub fn multi_miller_loop_y_q_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::YQC1,
            instance: i,
        })
    }
    pub fn multi_miller_loop_is_double(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::IsDouble,
            instance: i,
        })
    }

    // --- Frobenius ---
    pub fn frobenius_input(i: usize) -> Self {
        Self::Recursion(RecursionPoly::Frobenius {
            term: FrobeniusTerm::Input,
            instance: i,
        })
    }
    pub fn frobenius_output(i: usize) -> Self {
        Self::Recursion(RecursionPoly::Frobenius {
            term: FrobeniusTerm::Output,
            instance: i,
        })
    }
    pub fn frobenius_frob_const(i: usize) -> Self {
        Self::Recursion(RecursionPoly::Frobenius {
            term: FrobeniusTerm::FrobConst,
            instance: i,
        })
    }

    pub fn multi_miller_loop_is_add(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::IsAdd,
            instance: i,
        })
    }
    pub fn multi_miller_loop_l_val(i: usize) -> Self {
        Self::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::LVal,
            instance: i,
        })
    }

    // --- G1 addition ---
    pub fn g1_add_fused(term: G1AddTerm) -> Self {
        Self::Recursion(RecursionPoly::G1AddFused { term })
    }
    pub fn g1_add_xp(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::XP,
            instance: i,
        })
    }
    pub fn g1_add_yp(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::YP,
            instance: i,
        })
    }
    pub fn g1_add_p_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::PIndicator,
            instance: i,
        })
    }
    pub fn g1_add_xq(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::XQ,
            instance: i,
        })
    }
    pub fn g1_add_yq(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::YQ,
            instance: i,
        })
    }
    pub fn g1_add_q_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::QIndicator,
            instance: i,
        })
    }
    pub fn g1_add_xr(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::XR,
            instance: i,
        })
    }
    pub fn g1_add_yr(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::YR,
            instance: i,
        })
    }
    pub fn g1_add_r_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::RIndicator,
            instance: i,
        })
    }
    pub fn g1_add_lambda(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::Lambda,
            instance: i,
        })
    }
    pub fn g1_add_inv_delta_x(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::InvDeltaX,
            instance: i,
        })
    }
    pub fn g1_add_is_double(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::IsDouble,
            instance: i,
        })
    }
    pub fn g1_add_is_inverse(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1Add {
            term: G1AddTerm::IsInverse,
            instance: i,
        })
    }

    // --- G1 scalar multiplication ---
    pub fn g1_scalar_mul_xa(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::XA,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_ya(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::YA,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_xt(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::XT,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_yt(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::YT,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_xa_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::XANext,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_ya_next(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::YANext,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_t_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::TIndicator,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_a_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::AIndicator,
            instance: i,
        })
    }
    pub fn g1_scalar_mul_bit(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::Bit,
            instance: i,
        })
    }

    // --- G2 addition ---
    pub fn g2_add_xp_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::XPC0,
            instance: i,
        })
    }
    pub fn g2_add_xp_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::XPC1,
            instance: i,
        })
    }
    pub fn g2_add_yp_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::YPC0,
            instance: i,
        })
    }
    pub fn g2_add_yp_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::YPC1,
            instance: i,
        })
    }
    pub fn g2_add_p_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::PIndicator,
            instance: i,
        })
    }
    pub fn g2_add_xq_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::XQC0,
            instance: i,
        })
    }
    pub fn g2_add_xq_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::XQC1,
            instance: i,
        })
    }
    pub fn g2_add_yq_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::YQC0,
            instance: i,
        })
    }
    pub fn g2_add_yq_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::YQC1,
            instance: i,
        })
    }
    pub fn g2_add_q_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::QIndicator,
            instance: i,
        })
    }
    pub fn g2_add_xr_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::XRC0,
            instance: i,
        })
    }
    pub fn g2_add_xr_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::XRC1,
            instance: i,
        })
    }
    pub fn g2_add_yr_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::YRC0,
            instance: i,
        })
    }
    pub fn g2_add_yr_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::YRC1,
            instance: i,
        })
    }
    pub fn g2_add_r_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::RIndicator,
            instance: i,
        })
    }
    pub fn g2_add_lambda_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::LambdaC0,
            instance: i,
        })
    }
    pub fn g2_add_lambda_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::LambdaC1,
            instance: i,
        })
    }
    pub fn g2_add_inv_delta_x_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::InvDeltaXC0,
            instance: i,
        })
    }
    pub fn g2_add_inv_delta_x_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::InvDeltaXC1,
            instance: i,
        })
    }
    pub fn g2_add_is_double(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::IsDouble,
            instance: i,
        })
    }
    pub fn g2_add_is_inverse(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2Add {
            term: G2AddTerm::IsInverse,
            instance: i,
        })
    }

    // --- G2 scalar multiplication ---
    pub fn g2_scalar_mul_xa_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::XAC0,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_xa_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::XAC1,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_ya_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::YAC0,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_ya_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::YAC1,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_xt_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::XTC0,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_xt_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::XTC1,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_yt_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::YTC0,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_yt_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::YTC1,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_xa_next_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::XANextC0,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_xa_next_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::XANextC1,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_ya_next_c0(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::YANextC0,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_ya_next_c1(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::YANextC1,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_t_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::TIndicator,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_a_indicator(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::AIndicator,
            instance: i,
        })
    }
    pub fn g2_scalar_mul_bit(i: usize) -> Self {
        Self::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::Bit,
            instance: i,
        })
    }
}
