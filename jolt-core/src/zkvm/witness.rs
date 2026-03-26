#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    zkvm::ram::remap_address,
};

use super::instruction::{CircuitFlags, LookupQuery};

/// Compute the unsigned register increment for a cycle: `rd_inc + 2^XLEN`.
pub(super) fn rd_unsigned_inc(cycle: &Cycle) -> u128 {
    let (_, rd_pre, rd_post) = cycle.rd_write().unwrap_or_default();
    let rd_inc = rd_post as i128 - rd_pre as i128;
    (rd_inc + (1i128 << XLEN)) as u128
}

/// Compute the unsigned RAM increment for a cycle: `ram_inc + 2^XLEN`.
pub(super) fn ram_unsigned_inc(cycle: &Cycle) -> u128 {
    let ram_inc = match cycle.ram_access() {
        tracer::instruction::RAMAccess::Write(write) => {
            write.post_value as i128 - write.pre_value as i128
        }
        _ => 0,
    };
    (ram_inc + (1i128 << XLEN)) as u128
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /// Dense increment polynomial for registers (Dory path).
    RdInc,
    /// Dense increment polynomial for RAM (Dory path).
    RamInc,
    /// One-hot ra polynomial for instruction lookups (Shout).
    InstructionRa(usize),
    /// One-hot ra polynomial for bytecode (Shout).
    BytecodeRa(usize),
    /// One-hot ra/wa polynomial for RAM (Twist).
    RamRa(usize),
    /// One-hot ra polynomial for register increment (Hachi path).
    /// 8 polynomials `RdIncRa(0)..RdIncRa(7)` encoding the lower 64 bits of
    /// `unsigned_rd_inc = rd_inc + 2^XLEN` as chunks 1..d_inc-1.
    RdIncRa(usize),
    /// One-hot polynomial for bit 64 (MSB) of unsigned_rd_inc (Hachi path).
    /// Committed as OneHot(K=256) with indices 0 or 1 for uniform treatment.
    RdIncMsb,
    /// One-hot ra polynomial for RAM increment (Hachi path).
    /// 8 polynomials `RamIncRa(0)..RamIncRa(7)` encoding the lower 64 bits of
    /// `unsigned_ram_inc = ram_inc + 2^XLEN` as chunks 1..d_inc-1.
    RamIncRa(usize),
    /// One-hot polynomial for bit 64 (MSB) of unsigned_ram_inc (Hachi path).
    /// Committed as OneHot(K=256) with indices 0 or 1 for uniform treatment.
    RamIncMsb,
    /// Trusted advice polynomial.
    TrustedAdvice,
    /// Untrusted advice polynomial.
    UntrustedAdvice,
}

/// Returns the list of committed polynomials for a given PCS.
///
/// When `onehot_inc` is false (Dory), increments are committed as dense `RdInc` / `RamInc`.
/// When `onehot_inc` is true (Hachi), increments are committed as:
/// - `RdIncRa(0..7)` + `RdIncMsb`
/// - `RamIncRa(0..7)` + `RamIncMsb`
pub fn all_committed_polynomials(
    one_hot_params: &OneHotParams,
    onehot_inc: bool,
) -> Vec<CommittedPolynomial> {
    let mut polynomials = Vec::new();

    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::InstructionRa(i));
    }

    if onehot_inc {
        for i in 0..one_hot_params.inc_onehot_d() {
            polynomials.push(CommittedPolynomial::RdIncRa(i));
        }
        for i in 0..one_hot_params.inc_onehot_d() {
            polynomials.push(CommittedPolynomial::RamIncRa(i));
        }
    } else {
        polynomials.push(CommittedPolynomial::RdInc);
        polynomials.push(CommittedPolynomial::RamInc);
    }

    for i in 0..one_hot_params.ram_d {
        polynomials.push(CommittedPolynomial::RamRa(i));
    }
    for i in 0..one_hot_params.bytecode_d {
        polynomials.push(CommittedPolynomial::BytecodeRa(i));
    }

    if onehot_inc {
        polynomials.push(CommittedPolynomial::RdIncMsb);
        polynomials.push(CommittedPolynomial::RamIncMsb);
    }

    polynomials
}

impl CommittedPolynomial {
    /// Generate witness data and compute tier 1 commitment for a single row
    pub fn stream_witness_and_commit_rows<F, PCS>(
        &self,
        setup: &PCS::ProverSetup,
        preprocessing: &JoltSharedPreprocessing,
        row_cycles: &[tracer::instruction::Cycle],
        one_hot_params: &OneHotParams,
    ) -> <PCS as StreamingCommitmentScheme>::ChunkState
    where
        F: JoltField,
        PCS: StreamingCommitmentScheme<Field = F>,
    {
        let pcs = PCS::default();
        match self {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                pcs.process_chunk(setup, &row)
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
                pcs.process_chunk(setup, &row)
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize)
                    })
                    .collect();
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = preprocessing.bytecode.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
                    })
                    .collect();
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
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
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::RdIncRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let unsigned_inc = rd_unsigned_inc(cycle);
                        // idx maps to chunks 1..d_inc-1 (skip chunk 0 = MSB)
                        Some(one_hot_params.inc_chunk(unsigned_inc, idx + 1) as usize)
                    })
                    .collect();
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::RdIncMsb => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| Some((rd_unsigned_inc(cycle) >> XLEN) as usize))
                    .collect();
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::RamIncRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let unsigned_inc = ram_unsigned_inc(cycle);
                        // idx maps to chunks 1..d_inc-1 (skip chunk 0 = MSB)
                        Some(one_hot_params.inc_chunk(unsigned_inc, idx + 1) as usize)
                    })
                    .collect();
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::RamIncMsb => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| Some((ram_unsigned_inc(cycle) >> XLEN) as usize))
                    .collect();
                pcs.process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not use streaming witness generation")
            }
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
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
                        let pc = bytecode_preprocessing.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *i))
                    })
                    .collect();
                let t = addresses.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                    t,
                ))
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
                let t = addresses.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                    t,
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
                let t = addresses.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                    t,
                ))
            }
            CommittedPolynomial::RdIncRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let unsigned_inc = rd_unsigned_inc(cycle);
                        Some(one_hot_params.inc_chunk(unsigned_inc, i + 1))
                    })
                    .collect();
                let t = addresses.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                    t,
                ))
            }
            CommittedPolynomial::RdIncMsb => {
                let one_hot_params = one_hot_params.unwrap();
                let indices: Vec<Option<u8>> = trace
                    .par_iter()
                    .map(|cycle| Some((rd_unsigned_inc(cycle) >> XLEN) as u8))
                    .collect();
                let t = indices.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    indices,
                    one_hot_params.k_chunk,
                    t,
                ))
            }
            CommittedPolynomial::RamIncRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let unsigned_inc = ram_unsigned_inc(cycle);
                        Some(one_hot_params.inc_chunk(unsigned_inc, i + 1))
                    })
                    .collect();
                let t = addresses.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                    t,
                ))
            }
            CommittedPolynomial::RamIncMsb => {
                let one_hot_params = one_hot_params.unwrap();
                let indices: Vec<Option<u8>> = trace
                    .par_iter()
                    .map(|cycle| Some((ram_unsigned_inc(cycle) >> XLEN) as u8))
                    .collect();
                let t = indices.len();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    indices,
                    one_hot_params.k_chunk,
                    t,
                ))
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not use generate_witness")
            }
        }
    }

    #[inline(always)]
    pub fn extract_index(
        &self,
        cycle: &Cycle,
        preprocessing: &JoltSharedPreprocessing,
        one_hot_params: &OneHotParams,
    ) -> Option<u8> {
        match self {
            CommittedPolynomial::InstructionRa(i) => {
                let idx = LookupQuery::<XLEN>::to_lookup_index(cycle);
                Some(one_hot_params.lookup_index_chunk(idx, *i))
            }
            CommittedPolynomial::BytecodeRa(i) => {
                let pc = preprocessing.bytecode.get_pc(cycle);
                Some(one_hot_params.bytecode_pc_chunk(pc, *i))
            }
            CommittedPolynomial::RamRa(i) => remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.memory_layout,
            )
            .map(|addr| one_hot_params.ram_address_chunk(addr, *i)),
            CommittedPolynomial::RdIncRa(i) => {
                Some(one_hot_params.inc_chunk(rd_unsigned_inc(cycle), i + 1))
            }
            CommittedPolynomial::RdIncMsb => Some((rd_unsigned_inc(cycle) >> XLEN) as u8),
            CommittedPolynomial::RamIncRa(i) => {
                Some(one_hot_params.inc_chunk(ram_unsigned_inc(cycle), i + 1))
            }
            CommittedPolynomial::RamIncMsb => Some((ram_unsigned_inc(cycle) >> XLEN) as u8),
            _ => panic!("extract_index called on non-onehot polynomial"),
        }
    }

    pub fn get_onehot_k(&self, one_hot_params: &OneHotParams) -> Option<usize> {
        match self {
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_)
            | CommittedPolynomial::RdIncRa(_)
            | CommittedPolynomial::RamIncRa(_)
            | CommittedPolynomial::RdIncMsb
            | CommittedPolynomial::RamIncMsb => Some(one_hot_params.k_chunk),
            _ => None,
        }
    }
}

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
}
