//! Compile-time constant R1CS constraints with grouped evaluation
//!
//! This module provides a static, compile-time representation of R1CS constraints
//! and evaluates them in two groups optimized for the univariate-skip protocol.
//! Group 0 and Group 1 are evaluated separately and then folded via Lagrange
//! weights using fused accumulators.
//!
//! - Group 0 (first group) contains `UNIVARIATE_SKIP_DOMAIN_SIZE = ceil(N/2)`
//!   constraints. Its `Az` are booleans and its `Bz` fit in `i128`.
//! - Group 1 (second group) is the complement of Group 0. Its `Az` are `u8`
//!   and its `Bz` use `S160` for wider arithmetic.
//!
//! Grouped evaluation entry points:
//! - `eval_az_first_group` -> `[bool; UNIVARIATE_SKIP_DOMAIN_SIZE]`
//! - `eval_bz_first_group` -> `[i128; UNIVARIATE_SKIP_DOMAIN_SIZE]`
//! - `eval_az_second_group` -> `[u8; NUM_REMAINING_R1CS_CONSTRAINTS]`
//! - `eval_bz_second_group` -> `[S160; NUM_REMAINING_R1CS_CONSTRAINTS]`
//! - `compute_az_r_group0/group1` and `compute_bz_r_group0/group1` fold these
//!   vectors against Lagrange weights at the evaluation point `r` using
//!   specialized fused accumulators with a single Barrett reduction at the end.
//!
//! ## Adding a new constraint
//!
//! 1. Add a new variant to `ConstraintName` (maintain the same order as `UNIFORM_R1CS`).
//! 2. Add the constraint to `UNIFORM_R1CS` using the appropriate macro.
//! 3. Assign the constraint to a group:
//!    - Put its name in `UNIFORM_R1CS_FIRST_GROUP_NAMES` if it fits Group 0
//!      characteristics (boolean guards, ~64-bit differences in `Bz`).
//!    - Otherwise it will appear in Group 1 automatically as the complement.
//! 4. Maintain the grouping invariant: the first group must contain exactly
//!    `UNIVARIATE_SKIP_DOMAIN_SIZE = ceil(NUM_R1CS_CONSTRAINTS/2)` constraints,
//!    so the first group never has fewer elements than the second.
//!
//! ## Removing a constraint
//!
//! 1. Remove it from `UNIFORM_R1CS`.
//! 2. Remove the corresponding variant from `ConstraintName`.
//! 3. If present, remove its name from `UNIFORM_R1CS_FIRST_GROUP_NAMES`.
//! 4. Re-check that `UNIFORM_R1CS_FIRST_GROUP_NAMES.len()` equals
//!    `UNIVARIATE_SKIP_DOMAIN_SIZE` after the change; adjust the first group
//!    selection to satisfy the invariant that the first group is never smaller
//!    than the second.
//!
//! ## Grouping guidance
//!
//! - Prefer Group 0 for boolean `Az` and `Bz` that can be expressed as `i128`.
//! - Prefer Group 1 for constraints whose `Az` are small nonnegative integers
//!   and whose `Bz` require wider arithmetic (`S160`).
//! - This split minimizes conversions and maximizes accumulator efficiency.

use super::inputs::{JoltR1CSInputs, R1CSCycleInputs};
use crate::field::{AccumulateInPlace, JoltField};
use crate::utils::accumulation::{Acc5U, Acc6S, Acc7S};
use crate::zkvm::instruction::CircuitFlags;
use ark_ff::biginteger::{S128, S160, S192, S64};
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter};

pub use super::ops::{Term, LC};

/// A single R1CS constraint row
#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    pub a: LC,
    pub b: LC,
    // No c needed for now, all eq-conditional constraints
    // pub c: LC,
}

impl Constraint {
    pub const fn new(a: LC, b: LC) -> Self {
        Self { a, b }
    }
}

/// Creates: condition * (left - right) == 0
pub const fn constraint_eq_conditional_lc(condition: LC, left: LC, right: LC) -> Constraint {
    Constraint::new(
        condition,
        match left.checked_sub(right) {
            Some(b) => b,
            None => LC::zero(),
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, EnumCount, EnumIter)]
pub enum ConstraintName {
    RamAddrEqRs1PlusImmIfLoadStore,
    RamAddrEqZeroIfNotLoadStore,
    RamReadEqRamWriteIfLoad,
    RamReadEqRdWriteIfLoad,
    Rs2EqRamWriteIfStore,
    LeftLookupZeroUnlessAddSubMul,
    LeftLookupEqLeftInputOtherwise,
    RightLookupAdd,
    RightLookupSub,
    RightLookupEqProductIfMul,
    RightLookupEqRightInputOtherwise,
    AssertLookupOne,
    RdWriteEqLookupIfWriteLookupToRd,
    RdWriteEqPCPlusConstIfWritePCtoRD,
    NextUnexpPCEqLookupIfShouldJump,
    NextUnexpPCEqPCPlusImmIfShouldBranch,
    NextUnexpPCUpdateOtherwise,
    NextPCEqPCPlusOneIfInline,
    MustStartSequenceFromBeginning,
}

#[derive(Clone, Copy, Debug)]
pub struct NamedConstraint {
    pub name: ConstraintName,
    pub cons: Constraint,
}

/// r1cs_eq_conditional!: verbose, condition-first equality constraint
///
/// Usage: `r1cs_eq_conditional!(name: ConstraintName::Foo, if { COND } => { LEFT } == { RIGHT });`
#[macro_export]
macro_rules! r1cs_eq_conditional {
    (name: $nm:expr, if { $($cond:tt)* } => ( $($left:tt)* ) == ( $($right:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedConstraint {
            name: $nm,
            cons: $crate::zkvm::r1cs::constraints::constraint_eq_conditional_lc(
                $crate::lc!($($cond)*),
                $crate::lc!($($left)*),
                $crate::lc!($($right)*),
            ),
        }
    }};
}

/// Number of uniform R1CS constraints
pub const NUM_R1CS_CONSTRAINTS: usize = ConstraintName::COUNT;

/// Static table of all R1CS uniform constraints.
pub static UNIFORM_R1CS: [NamedConstraint; NUM_R1CS_CONSTRAINTS] = [
    // if Load || Store {
    //     assert!(RamAddress == Rs1Value + Imm)
    // } else {
    //     assert!(RamAddress == 0)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RamAddrEqRs1PlusImmIfLoadStore,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::RamAddress } ) == ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::RamAddrEqZeroIfNotLoadStore,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::RamAddress } ) == ( { 0i128 } )
    ),
    // if Load {
    //     assert!(RamReadValue == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RamReadEqRamWriteIfLoad,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),
    // if Load {
    //     assert!(RamReadValue == RdWriteValue)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RamReadEqRdWriteIfLoad,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RdWriteValue } )
    ),
    // if Store {
    //     assert!(Rs2Value == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::Rs2EqRamWriteIfStore,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::Rs2Value } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),
    // if AddOperands || SubtractOperands || MultiplyOperands {
    //     // Lookup query is just RightLookupOperand
    //     assert!(LeftLookupOperand == 0)
    // } else {
    //     assert!(LeftLookupOperand == LeftInstructionInput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::LeftLookupZeroUnlessAddSubMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::LeftLookupOperand } ) == ( { 0i128 } )
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::LeftLookupEqLeftInputOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::LeftLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } )
    ),
    // If AddOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupAdd,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } + { JoltR1CSInputs::RightInstructionInput } )
    ),
    // If SubtractOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
    // }
    // Converts from unsigned to twos-complement representation
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupSub,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } - { JoltR1CSInputs::RightInstructionInput } + { 0x10000000000000000i128 } )
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupEqProductIfMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::Product } )
    ),
    // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
    //     assert!(RightLookupOperand == RightInstructionInput)
    // }
    // Arbitrary untrusted advice goes in right lookup operand
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupEqRightInputOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Advice) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::RightInstructionInput } )
    ),
    // if Assert {
    //     assert!(LookupOutput == 1)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::AssertLookupOne,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Assert) } }
        => ( { JoltR1CSInputs::LookupOutput } ) == ( { 1i128 } )
    ),
    // if Rd != 0 && WriteLookupOutputToRD {
    //     assert!(RdWriteValue == LookupOutput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RdWriteEqLookupIfWriteLookupToRd,
        if { { JoltR1CSInputs::WriteLookupOutputToRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Rd != 0 && Jump {
    //     if !isCompressed {
    //          assert!(RdWriteValue == UnexpandedPC + 4)
    //     } else {
    //          assert!(RdWriteValue == UnexpandedPC + 2)
    //     }
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD,
        if { { JoltR1CSInputs::WritePCtoRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Jump && !NextIsNoop {
    //     assert!(NextUnexpandedPC == LookupOutput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCEqLookupIfShouldJump,
        if { { JoltR1CSInputs::ShouldJump } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Branch && LookupOutput {
    //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch,
        if { { JoltR1CSInputs::ShouldBranch } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::UnexpandedPC } + { JoltR1CSInputs::Imm } )
    ),
    // if !(ShouldBranch || Jump) {
    //     if DoNotUpdatePC {
    //         assert!(NextUnexpandedPC == UnexpandedPC)
    //     } else if isCompressed {
    //         assert!(NextUnexpandedPC == UnexpandedPC + 2)
    //     } else {
    //         assert!(NextUnexpandedPC == UnexpandedPC + 4)
    //     }
    // }
    // Note that ShouldBranch and Jump instructions are mutually exclusive
    // And that DoNotUpdatePC and isCompressed are mutually exclusive
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCUpdateOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::ShouldBranch } - { JoltR1CSInputs::OpFlags(CircuitFlags::Jump) } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } )
           == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 }
                - { 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) }
                - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Inline {
    //     assert!(NextPC == PC + 1)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::NextPCEqPCPlusOneIfInline,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction) } }
        => ( { JoltR1CSInputs::NextPC } ) == ( { JoltR1CSInputs::PC } + { 1i128 } )
    ),
    // if NextIsVirtual && !NextIsFirstInSequence {
    //     assert!(1 == DoNotUpdateUnexpandedPC)
    // }
    // (we write this way so that 1 - flag is boolean)
    r1cs_eq_conditional!(
        name: ConstraintName::MustStartSequenceFromBeginning,
        if { { JoltR1CSInputs::NextIsVirtual } - { JoltR1CSInputs::NextIsFirstInSequence } }
        => ( { 1 } ) == ( { JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) } )
    ),
];

// =============================================================================
// Univariate skip constants and grouped views
// =============================================================================

/// Degree of univariate skip, defined to be `(NUM_R1CS_CONSTRAINTS - 1) / 2`
pub const UNIVARIATE_SKIP_DEGREE: usize = (NUM_R1CS_CONSTRAINTS - 1) / 2;

/// Domain size of univariate skip, defined to be `UNIVARIATE_SKIP_DEGREE + 1`.
pub const UNIVARIATE_SKIP_DOMAIN_SIZE: usize = UNIVARIATE_SKIP_DEGREE + 1;

/// Extended domain size of univariate skip, defined to be `2 * UNIVARIATE_SKIP_DEGREE + 1`.
pub const UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize = 2 * UNIVARIATE_SKIP_DEGREE + 1;

/// Number of coefficients in the first-round polynomial, defined to be `3 * UNIVARIATE_SKIP_DEGREE + 1`.
pub const FIRST_ROUND_POLY_NUM_COEFFS: usize = 3 * UNIVARIATE_SKIP_DEGREE + 1;

/// Number of remaining R1CS constraints in the second group, defined to be
/// `NUM_R1CS_CONSTRAINTS - UNIVARIATE_SKIP_DOMAIN_SIZE`.
pub const NUM_REMAINING_R1CS_CONSTRAINTS: usize =
    NUM_R1CS_CONSTRAINTS - UNIVARIATE_SKIP_DOMAIN_SIZE;

/// Order-preserving, compile-time filter over `UNIFORM_R1CS` by constraint names.
const fn contains_name<const N: usize>(names: &[ConstraintName; N], name: ConstraintName) -> bool {
    let mut i = 0;
    while i < N {
        if names[i] as u32 == name as u32 {
            return true;
        }
        i += 1;
    }
    false
}

/// Select constraints from `UNIFORM_R1CS` whose names appear in `names`, preserving order.
pub const fn filter_uniform_r1cs<const N: usize>(
    names: &[ConstraintName; N],
) -> [NamedConstraint; N] {
    let dummy = NamedConstraint {
        name: ConstraintName::RamReadEqRamWriteIfLoad,
        cons: Constraint::new(LC::zero(), LC::zero()),
    };
    let mut out: [NamedConstraint; N] = [dummy; N];

    let mut o = 0;
    let mut i = 0;
    while i < NUM_R1CS_CONSTRAINTS {
        let cand = UNIFORM_R1CS[i];
        if contains_name(names, cand.name) {
            out[o] = cand;
            o += 1;
            if o == N {
                break;
            }
        }
        i += 1;
    }

    if o != N {
        panic!("filter_uniform_r1cs: not all requested constraints were found in UNIFORM_R1CS");
    }
    out
}

/// Compute the complement of `UNIFORM_R1CS_FIRST_GROUP_NAMES` within `UNIFORM_R1CS`.
const fn complement_first_group_names() -> [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] {
    let mut out: [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] =
        [ConstraintName::RamReadEqRamWriteIfLoad; NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut o = 0;
    let mut i = 0;
    while i < NUM_R1CS_CONSTRAINTS {
        let cand = UNIFORM_R1CS[i].name;
        if !contains_name(&UNIFORM_R1CS_FIRST_GROUP_NAMES, cand) {
            out[o] = cand;
            o += 1;
            if o == NUM_REMAINING_R1CS_CONSTRAINTS {
                break;
            }
        }
        i += 1;
    }

    if o != NUM_REMAINING_R1CS_CONSTRAINTS {
        panic!("complement_first_group_names: expected full complement");
    }
    out
}

/// First group: 10 boolean-guarded eq constraints, where Bz is around 64 bits
pub const UNIFORM_R1CS_FIRST_GROUP_NAMES: [ConstraintName; UNIVARIATE_SKIP_DOMAIN_SIZE] = [
    ConstraintName::RamAddrEqZeroIfNotLoadStore,
    ConstraintName::RamReadEqRamWriteIfLoad,
    ConstraintName::RamReadEqRdWriteIfLoad,
    ConstraintName::Rs2EqRamWriteIfStore,
    ConstraintName::LeftLookupZeroUnlessAddSubMul,
    ConstraintName::LeftLookupEqLeftInputOtherwise,
    ConstraintName::AssertLookupOne,
    ConstraintName::NextUnexpPCEqLookupIfShouldJump,
    ConstraintName::NextPCEqPCPlusOneIfInline,
    ConstraintName::MustStartSequenceFromBeginning,
];

/// Second group: complement of first within UNIFORM_R1CS
/// Here, Az may be u8, and Bz may be around 128 bits
pub const UNIFORM_R1CS_SECOND_GROUP_NAMES: [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] =
    complement_first_group_names();

/// First group: 10 boolean-guarded eq constraints, where Bz is around 64 bits
pub static UNIFORM_R1CS_FIRST_GROUP: [NamedConstraint; UNIVARIATE_SKIP_DOMAIN_SIZE] =
    filter_uniform_r1cs(&UNIFORM_R1CS_FIRST_GROUP_NAMES);

/// Second group: complement of first within UNIFORM_R1CS, where Az may be u8 and Bz may be around 128 bits
pub static UNIFORM_R1CS_SECOND_GROUP: [NamedConstraint; NUM_REMAINING_R1CS_CONSTRAINTS] =
    filter_uniform_r1cs(&UNIFORM_R1CS_SECOND_GROUP_NAMES);

/// Az evaluation for the first group - all boolean flags
#[derive(Clone, Copy, Debug)]
pub struct AzFirstGroup {
    pub ram_addr_eq_zero_if_not_load_store: bool,      // !(Load || Store)
    pub ram_read_eq_ram_write_if_load: bool,           // Load
    pub ram_read_eq_rd_write_if_load: bool,            // Load
    pub rs2_eq_ram_write_if_store: bool,               // Store
    pub left_lookup_zero_unless_add_sub_mul: bool,     // Add || Sub || Mul
    pub left_lookup_eq_left_input_otherwise: bool,     // !(Add || Sub || Mul)
    pub assert_lookup_one: bool,                       // Assert
    pub next_unexp_pc_eq_lookup_if_should_jump: bool,  // ShouldJump
    pub next_pc_eq_pc_plus_one_if_inline: bool,        // VirtualInstruction
    pub must_start_sequence_from_beginning: bool,      // NextIsVirtual && !NextIsFirstInSequence
}

/// Bz evaluation for the first group - up to S64
#[derive(Clone, Copy, Debug)]
pub struct BzFirstGroup {
    pub ram_addr: u64,                    // RamAddress - 0
    pub ram_read_minus_ram_write: S64,    // RamReadValue - RamWriteValue
    pub ram_read_minus_rd_write: S64,     // RamReadValue - RdWriteValue
    pub rs2_minus_ram_write: S64,         // Rs2Value - RamWriteValue
    pub left_lookup: u64,                 // LeftLookupOperand - 0
    pub left_lookup_minus_left_input: S64, // LeftLookupOperand - LeftInstructionInput
    pub lookup_output_minus_one: S64,     // LookupOutput - 1
    pub next_unexp_pc_minus_lookup_output: S64, // NextUnexpandedPC - LookupOutput
    pub next_pc_minus_pc_plus_one: S64,   // NextPC - (PC + 1)
    pub do_not_update_unexpanded_pc_minus_one: bool, // 1 - DoNotUpdateUnexpandedPC
}

/// Az evaluation for the second group - all bool except for two u8s
#[derive(Clone, Copy, Debug)]
pub struct AzSecondGroup {
    pub ram_addr_eq_rs1_plus_imm_if_load_store: bool, // Load || Store
    pub right_lookup_add: bool,                       // Add
    pub right_lookup_sub: bool,                       // Sub
    pub right_lookup_eq_product_if_mul: bool,         // Mul
    pub right_lookup_eq_right_input_otherwise: bool,  // !(Add || Sub || Mul || Advice)
    pub rd_write_eq_lookup_if_write_lookup_to_rd: u8, // write_lookup_output_to_rd_addr (Rd != 0 encoded)
    pub rd_write_eq_pc_plus_const_if_write_pc_to_rd: u8, // write_pc_to_rd_addr (Rd != 0 encoded)
    pub next_unexp_pc_eq_pc_plus_imm_if_should_branch: bool, // ShouldBranch
    pub next_unexp_pc_update_otherwise: bool,           // !(Jump || ShouldBranch)
}

/// Bz evaluation for the second group - mixed precision (up to S160) based on actual value ranges
#[derive(Clone, Copy, Debug)]
pub struct BzSecondGroup {
    pub ram_addr_minus_rs1_plus_imm: i128,            // RamAddress - (Rs1Value + Imm)
    pub right_lookup_minus_add_result: S160,          // RightLookupOperand - (LeftInput + RightInput)
    pub right_lookup_minus_sub_result: S160,          // RightLookupOperand - (LeftInput - RightInput + 2^64)
    pub right_lookup_minus_product: S160,             // RightLookupOperand - Product
    pub right_lookup_minus_right_input: S160,         // RightLookupOperand - RightInput
    pub rd_write_minus_lookup_output: S64,            // RdWriteValue - LookupOutput
    pub rd_write_minus_pc_plus_const: i128,            // RdWriteValue - (UnexpandedPC + const)
    pub next_unexp_pc_minus_pc_plus_imm: i128,        // NextUnexpandedPC - (UnexpandedPC + Imm)
    pub next_unexp_pc_minus_expected: i128,           // NextUnexpandedPC - (UnexpandedPC + const)
}

/// A convenient struct for evaluating the first group of constraints in the univariate skip
/// and streaming round
#[derive(Clone, Copy, Debug)]
pub struct R1CSFirstGroup<'a, F: JoltField> {
    row: &'a R1CSCycleInputs,
    _m: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField> R1CSFirstGroup<'a, F> {
    #[inline(always)]
    pub fn from_cycle_inputs(row: &'a R1CSCycleInputs) -> Self {
        Self {
            row,
            _m: core::marker::PhantomData,
        }
    }

    /// Evaluate Az for the first group
    #[inline]
    pub fn eval_az(&self) -> AzFirstGroup {
        let flags = &self.row.flags;
        let ld = flags[CircuitFlags::Load];
        let st = flags[CircuitFlags::Store];
        let add = flags[CircuitFlags::AddOperands];
        let sub = flags[CircuitFlags::SubtractOperands];
        let mul = flags[CircuitFlags::MultiplyOperands];
        let assert_flag = flags[CircuitFlags::Assert];
        let inline_seq = flags[CircuitFlags::VirtualInstruction];

        AzFirstGroup {
            ram_addr_eq_zero_if_not_load_store: !(ld || st),
            ram_read_eq_ram_write_if_load: ld,
            ram_read_eq_rd_write_if_load: ld,
            rs2_eq_ram_write_if_store: st,
            left_lookup_zero_unless_add_sub_mul: add || sub || mul,
            left_lookup_eq_left_input_otherwise: !(add || sub || mul),
            assert_lookup_one: assert_flag,
            next_unexp_pc_eq_lookup_if_should_jump: self.row.should_jump,
            next_pc_eq_pc_plus_one_if_inline: inline_seq,
            must_start_sequence_from_beginning:
            self.row.next_is_virtual && !self.row.next_is_first_in_sequence,
        }
    }

    /// Evaluate Bz for the first group
    #[inline]
    pub fn eval_bz(&self) -> BzFirstGroup {
        let left_lookup_u64 = self.row.left_lookup;
        let left_input_u64 = self.row.left_input;
        let ram_read_u64 = self.row.ram_read_value;
        let ram_write_u64 = self.row.ram_write_value;
        let rd_write_u64 = self.row.rd_write_value;
        let rs2_u64 = self.row.rs2_read_value;
        let ram_addr_u64 = self.row.ram_addr;
        let lookup_out_u64 = self.row.lookup_output;
        let next_unexp_pc_u64 = self.row.next_unexpanded_pc;
        let pc_u64 = self.row.pc;
        let next_pc_u64 = self.row.next_pc;
        let do_not_update_unexpanded_pc_flag = self.row.flags[CircuitFlags::DoNotUpdateUnexpandedPC];

        BzFirstGroup {
            // u64 values
            ram_addr: ram_addr_u64,
            left_lookup: left_lookup_u64,

            // signed 64-bit differences as S64
            ram_read_minus_ram_write: crate::utils::accumulation::s64_from_diff_u64s(ram_read_u64, ram_write_u64),
            ram_read_minus_rd_write: crate::utils::accumulation::s64_from_diff_u64s(ram_read_u64, rd_write_u64),
            rs2_minus_ram_write: crate::utils::accumulation::s64_from_diff_u64s(rs2_u64, ram_write_u64),
            left_lookup_minus_left_input: crate::utils::accumulation::s64_from_diff_u64s(left_lookup_u64, left_input_u64),
            lookup_output_minus_one: crate::utils::accumulation::s64_from_diff_u64s(lookup_out_u64, 1),
            next_unexp_pc_minus_lookup_output: crate::utils::accumulation::s64_from_diff_u64s(next_unexp_pc_u64, lookup_out_u64),
            next_pc_minus_pc_plus_one: crate::utils::accumulation::s64_from_diff_u64s(next_pc_u64, pc_u64.wrapping_add(1)),

            // boolean: 1 - DoNotUpdateUnexpandedPC
            do_not_update_unexpanded_pc_minus_one: !do_not_update_unexpanded_pc_flag,
        }
    }

    /// Product-of-sums for univariate-skip shift coefficients over the base window (Group 0)
    /// Computes: (sum_i c_i * Az_i) * (sum_i c_i * Bz_i) with Az in {0,1}, Bz as i128
    #[inline]
    pub fn product_of_sums_shifted(&self,
        coeffs_i32: &[i32; UNIVARIATE_SKIP_DOMAIN_SIZE],
        _coeffs_s64: &[S64; UNIVARIATE_SKIP_DOMAIN_SIZE],
    ) -> S192 {
        let az = self.eval_az();
        let bz = self.eval_bz();

        let mut sum_c_az_i64: i64 = 0;
        let mut sum_c_bz_s128 = S128::from(0i128);

        // Index 0
        let c0 = coeffs_i32[0] as i64;
        if az.ram_addr_eq_zero_if_not_load_store {
            sum_c_az_i64 += c0;
        }
        sum_c_bz_s128 += S128::from_i128(c0 as i128 * bz.ram_addr as i128);

        // 1
        let c1 = coeffs_i32[1] as i64;
        if az.ram_read_eq_ram_write_if_load {
            sum_c_az_i64 += c1;
        }
        sum_c_bz_s128 += S128::from_i128(c1 as i128 * bz.ram_read_minus_ram_write.to_i128());

        // 2
        let c2 = coeffs_i32[2] as i64;
        if az.ram_read_eq_rd_write_if_load {
            sum_c_az_i64 += c2;
        }
        sum_c_bz_s128 += S128::from_i128(c2 as i128 * bz.ram_read_minus_rd_write.to_i128());

        // 3
        let c3 = coeffs_i32[3] as i64;
        if az.rs2_eq_ram_write_if_store {
            sum_c_az_i64 += c3;
        }
        sum_c_bz_s128 += S128::from_i128(c3 as i128 * bz.rs2_minus_ram_write.to_i128());

        // 4
        let c4 = coeffs_i32[4] as i64;
        if az.left_lookup_zero_unless_add_sub_mul {
            sum_c_az_i64 += c4;
        }
        sum_c_bz_s128 += S128::from_i128(c4 as i128 * bz.left_lookup as i128);

        // 5
        let c5 = coeffs_i32[5] as i64;
        if az.left_lookup_eq_left_input_otherwise {
            sum_c_az_i64 += c5;
        }
        sum_c_bz_s128 += S128::from_i128(c5 as i128 * bz.left_lookup_minus_left_input.to_i128());

        // 6
        let c6 = coeffs_i32[6] as i64;
        if az.assert_lookup_one {
            sum_c_az_i64 += c6;
        }
        sum_c_bz_s128 += S128::from_i128(c6 as i128 * bz.lookup_output_minus_one.to_i128());

        // 7
        let c7 = coeffs_i32[7] as i64;
        if az.next_unexp_pc_eq_lookup_if_should_jump {
            sum_c_az_i64 += c7;
        }
        sum_c_bz_s128 += S128::from_i128(c7 as i128 * bz.next_unexp_pc_minus_lookup_output.to_i128());

        // 8
        let c8 = coeffs_i32[8] as i64;
        if az.next_pc_eq_pc_plus_one_if_inline {
            sum_c_az_i64 += c8;
        }
        sum_c_bz_s128 += S128::from_i128(c8 as i128 * bz.next_pc_minus_pc_plus_one.to_i128());

        // 9
        let c9 = coeffs_i32[9] as i64;
        if az.must_start_sequence_from_beginning {
            sum_c_az_i64 += c9;
        }
        // boolean term (1 - DoNotUpdateUnexpandedPC)
        sum_c_bz_s128 += S128::from_i128(c9 as i128 * (bz.do_not_update_unexpanded_pc_minus_one as i128));

        let sum_az_s64 = S64::from_i64(sum_c_az_i64);
        sum_az_s64.mul_trunc::<2, 3>(&sum_c_bz_s128)
    }

    /// Lagrange-folded Az at r0 using base-domain weights
    #[inline]
    pub fn az_at_r(&self, w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let az = self.eval_az();
        let mut acc: Acc5U<F> = Acc5U::new();
        acc.fmadd(&w[0], &az.ram_addr_eq_zero_if_not_load_store);
        acc.fmadd(&w[1], &az.ram_read_eq_ram_write_if_load);
        acc.fmadd(&w[2], &az.ram_read_eq_rd_write_if_load);
        acc.fmadd(&w[3], &az.rs2_eq_ram_write_if_store);
        acc.fmadd(&w[4], &az.left_lookup_zero_unless_add_sub_mul);
        acc.fmadd(&w[5], &az.left_lookup_eq_left_input_otherwise);
        acc.fmadd(&w[6], &az.assert_lookup_one);
        acc.fmadd(&w[7], &az.next_unexp_pc_eq_lookup_if_should_jump);
        acc.fmadd(&w[8], &az.next_pc_eq_pc_plus_one_if_inline);
        acc.fmadd(&w[9], &az.must_start_sequence_from_beginning);
        acc.reduce()
    }

    /// Lagrange-folded Bz at r0 using base-domain weights
    #[inline]
    pub fn bz_at_r(&self, w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let bz = self.eval_bz();
        let mut acc: Acc6S<F> = Acc6S::new();
        // TODO: define fmadd instance for Acc6S with bool/u64/S64
        let ram_addr_i128 = bz.ram_addr as i128;
        acc.fmadd(&w[0], &ram_addr_i128);
        let rr_rw_i128 = bz.ram_read_minus_ram_write.to_i128();
        acc.fmadd(&w[1], &rr_rw_i128);
        let rr_rd_i128 = bz.ram_read_minus_rd_write.to_i128();
        acc.fmadd(&w[2], &rr_rd_i128);
        let rs_rw_i128 = bz.rs2_minus_ram_write.to_i128();
        acc.fmadd(&w[3], &rs_rw_i128);
        let left_lookup_i128 = bz.left_lookup as i128;
        acc.fmadd(&w[4], &left_lookup_i128);
        let ll_li_i128 = bz.left_lookup_minus_left_input.to_i128();
        acc.fmadd(&w[5], &ll_li_i128);
        let lo_minus_one_i128 = bz.lookup_output_minus_one.to_i128();
        acc.fmadd(&w[6], &lo_minus_one_i128);
        let npc_minus_lo_i128 = bz.next_unexp_pc_minus_lookup_output.to_i128();
        acc.fmadd(&w[7], &npc_minus_lo_i128);
        let nextpc_minus_pc1_i128 = bz.next_pc_minus_pc_plus_one.to_i128();
        acc.fmadd(&w[8], &nextpc_minus_pc1_i128);
        acc.fmadd(&w[9], &bz.do_not_update_unexpanded_pc_minus_one);
        acc.reduce()
    }
}

impl<'a, F: JoltField> R1CSFirstGroup<'a, F> {
    #[cfg(test)]
    #[inline]
    pub fn debug_assert_zero_when_guarded(&self) {
        let az = self.eval_az();
        let bz = self.eval_bz();
        debug_assert!((!az.ram_addr_eq_zero_if_not_load_store) || bz.ram_addr == 0);
        debug_assert!((!az.ram_read_eq_ram_write_if_load) || bz.ram_read_minus_ram_write.to_i128() == 0);
        debug_assert!((!az.ram_read_eq_rd_write_if_load) || bz.ram_read_minus_rd_write.to_i128() == 0);
        debug_assert!((!az.rs2_eq_ram_write_if_store) || bz.rs2_minus_ram_write.to_i128() == 0);
        debug_assert!((!az.left_lookup_zero_unless_add_sub_mul) || bz.left_lookup == 0);
        debug_assert!((!az.left_lookup_eq_left_input_otherwise) || bz.left_lookup_minus_left_input.to_i128() == 0);
        debug_assert!((!az.assert_lookup_one) || bz.lookup_output_minus_one.to_i128() == 0);
        debug_assert!((!az.next_unexp_pc_eq_lookup_if_should_jump) || bz.next_unexp_pc_minus_lookup_output.to_i128() == 0);
        debug_assert!((!az.next_pc_eq_pc_plus_one_if_inline) || bz.next_pc_minus_pc_plus_one.to_i128() == 0);
        debug_assert!((!az.must_start_sequence_from_beginning) || bz.do_not_update_unexpanded_pc_minus_one);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct R1CSSecondGroup<'a, F: JoltField> {
    row: &'a R1CSCycleInputs,
    _m: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField> R1CSSecondGroup<'a, F> {
    #[inline]
    pub fn from_cycle_inputs(row: &'a R1CSCycleInputs) -> Self {
        Self {
            row,
            _m: core::marker::PhantomData,
        }
    }

    /// Evaluate Az for the second group
    #[inline]
    pub fn eval_az(&self) -> AzSecondGroup {
        let flags = &self.row.flags;
        let not_add_sub_mul_advice =
                    !(flags[CircuitFlags::AddOperands]
                        || flags[CircuitFlags::SubtractOperands]
                        || flags[CircuitFlags::MultiplyOperands]
                || flags[CircuitFlags::Advice]) as u8;
        let next_update_otherwise = {
                    let jump = flags[CircuitFlags::Jump] as u8;
                    let should_branch = self.row.should_branch as u8;
                    #[cfg(test)]
                    {
                        if jump + should_branch > 1 {
                            panic!("jump and should_branch are both set");
                        }
                    }
            1u8.wrapping_sub(jump).wrapping_sub(should_branch) == 1
        };

        AzSecondGroup {
            ram_addr_eq_rs1_plus_imm_if_load_store:
                (flags[CircuitFlags::Load] || flags[CircuitFlags::Store]),
            right_lookup_add: flags[CircuitFlags::AddOperands],
            right_lookup_sub: flags[CircuitFlags::SubtractOperands],
            right_lookup_eq_product_if_mul: flags[CircuitFlags::MultiplyOperands],
            right_lookup_eq_right_input_otherwise: not_add_sub_mul_advice == 1,
            rd_write_eq_lookup_if_write_lookup_to_rd: self.row.write_lookup_output_to_rd_addr,
            rd_write_eq_pc_plus_const_if_write_pc_to_rd: self.row.write_pc_to_rd_addr,
            next_unexp_pc_eq_pc_plus_imm_if_should_branch: self.row.should_branch,
            next_unexp_pc_update_otherwise: next_update_otherwise,
        }
    }

    /// Evaluate Bz for the second group
    #[inline]
    pub fn eval_bz(&self) -> BzSecondGroup {
        // RamAddrEqRs1PlusImmIfLoadStore
        let expected_addr: i128 = if self.row.imm.is_positive {
            (self.row.rs1_read_value as u128 + self.row.imm.magnitude_as_u64() as u128) as i128
                    } else {
                        self.row.rs1_read_value as i128 - self.row.imm.magnitude_as_u64() as i128
                    };
        let ram_addr_minus_rs1_plus_imm = self.row.ram_addr as i128 - expected_addr;

        // RightLookupAdd / Sub / Product / RightInput
        let right_add_expected = (self.row.left_input as i128) + self.row.right_input.to_i128();
        let right_sub_expected = (self.row.left_input as i128)
            - self.row.right_input.to_i128()
            + (1i128 << 64);

        let right_lookup_minus_add_result =
            S160::from(self.row.right_lookup) - S160::from(right_add_expected);
        let right_lookup_minus_sub_result =
            S160::from(self.row.right_lookup) - S160::from(right_sub_expected);

        let right_lookup_minus_product =
            S160::from(self.row.right_lookup) - S160::from(self.row.product);
        let right_lookup_minus_right_input =
            S160::from(self.row.right_lookup) - S160::from(self.row.right_input);

        // Rd write checks (fit in i64 range by construction)
        let rd_write_minus_lookup_output = S64::from((self.row.rd_write_value as i128) - (self.row.lookup_output as i128));
        let const_term = 4
            - if self.row.flags[CircuitFlags::IsCompressed] { 2 } else { 0 };
        let rd_write_minus_pc_plus_const = (self.row.rd_write_value as i128)
            - ((self.row.unexpanded_pc as i128) + (const_term as i128));

        // Next unexpanded PC checks
        let next_unexp_pc_minus_pc_plus_imm = (self.row.next_unexpanded_pc as i128)
            - (self.row.unexpanded_pc as i128 + self.row.imm.to_i128());

        let const_term_next = 4
            - if self.row.flags[CircuitFlags::DoNotUpdateUnexpandedPC] { 4 } else { 0 }
            - if self.row.flags[CircuitFlags::IsCompressed] { 2 } else { 0 };
        let target_next = self.row.unexpanded_pc as i128 + const_term_next;
        let next_unexp_pc_minus_expected = self.row.next_unexpanded_pc as i128 - target_next;

        BzSecondGroup {
            ram_addr_minus_rs1_plus_imm,
            right_lookup_minus_add_result,
            right_lookup_minus_sub_result,
            right_lookup_minus_product,
            right_lookup_minus_right_input,
            rd_write_minus_lookup_output,
            rd_write_minus_pc_plus_const,
            next_unexp_pc_minus_pc_plus_imm,
            next_unexp_pc_minus_expected,
        }
    }

    /// Product-of-sums for univariate-skip shift coefficients (Group 1)
    /// Computes: (sum_i c_i * Az_i) * (sum_i c_i * Bz_i)
    /// with Az as u8 and Bz as S160, padded to base window length.
    #[inline]
    pub fn product_of_sums_shifted(&self,
        coeffs_i32: &[i32; UNIVARIATE_SKIP_DOMAIN_SIZE],
        _coeffs_s64: &[S64; UNIVARIATE_SKIP_DOMAIN_SIZE],
    ) -> S192 {
        #[cfg(test)]
        self.debug_assert_zero_when_guarded();

        let az = self.eval_az();
        let bz = self.eval_bz();

        // Materialize compact arrays in group order (length up to UNIVARIATE_SKIP_DOMAIN_SIZE)
        let g2_len = core::cmp::min(NUM_REMAINING_R1CS_CONSTRAINTS, UNIVARIATE_SKIP_DOMAIN_SIZE);
        let mut az_arr: [u8; UNIVARIATE_SKIP_DOMAIN_SIZE] = [0u8; UNIVARIATE_SKIP_DOMAIN_SIZE];
        let mut bz_arr: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] = [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];

        // 0..=8 mapping matches field order
        az_arr[0] = az.ram_addr_eq_rs1_plus_imm_if_load_store as u8;
        bz_arr[0] = S160::from(bz.ram_addr_minus_rs1_plus_imm);

        az_arr[1] = az.right_lookup_add as u8;
        bz_arr[1] = bz.right_lookup_minus_add_result;

        az_arr[2] = az.right_lookup_sub as u8;
        bz_arr[2] = bz.right_lookup_minus_sub_result;

        az_arr[3] = az.right_lookup_eq_product_if_mul as u8;
        bz_arr[3] = bz.right_lookup_minus_product;

        az_arr[4] = az.right_lookup_eq_right_input_otherwise as u8;
        bz_arr[4] = bz.right_lookup_minus_right_input;

        az_arr[5] = az.rd_write_eq_lookup_if_write_lookup_to_rd;
        bz_arr[5] = S160::from(bz.rd_write_minus_lookup_output.to_i128());

        az_arr[6] = az.rd_write_eq_pc_plus_const_if_write_pc_to_rd;
        bz_arr[6] = S160::from(bz.rd_write_minus_pc_plus_const);

        az_arr[7] = az.next_unexp_pc_eq_pc_plus_imm_if_should_branch as u8;
        bz_arr[7] = S160::from(bz.next_unexp_pc_minus_pc_plus_imm);

        az_arr[8] = az.next_unexp_pc_update_otherwise as u8;
        bz_arr[8] = S160::from(bz.next_unexp_pc_minus_expected);

        let mut sum_c_az_i64: i64 = 0;
        let mut sum_bz_s160 = S160::from(0i128);

        let mut i = 0usize;
        while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
            let c_i64 = coeffs_i32[i] as i64;
            if i < g2_len {
                let az_i = az_arr[i] as i64;
                sum_c_az_i64 += c_i64.saturating_mul(az_i);
                let c_s160 = S160::from(c_i64 as i128);
                let term: S160 = (&c_s160) * (&bz_arr[i]);
                sum_bz_s160 += term;
            }
            i += 1;
        }

        let sum_bz_s192: S192 = sum_bz_s160.to_signed_bigint_nplus1::<3>();
        let sum_az_s64 = S64::from_i64(sum_c_az_i64);
        sum_az_s64.mul_trunc::<3, 3>(&sum_bz_s192)
    }

    /// Lagrange-folded Az at r0 using base-domain weights (iterate up to group-1 length)
    #[inline]
    pub fn az_at_r(&self, w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let az = self.eval_az();
        let mut acc: Acc5U<F> = Acc5U::new();
        acc.fmadd(&w[0], &az.ram_addr_eq_rs1_plus_imm_if_load_store);
        acc.fmadd(&w[1], &az.right_lookup_add);
        acc.fmadd(&w[2], &az.right_lookup_sub);
        acc.fmadd(&w[3], &az.right_lookup_eq_product_if_mul);
        acc.fmadd(&w[4], &az.right_lookup_eq_right_input_otherwise);
        acc.fmadd(&w[5], &az.rd_write_eq_lookup_if_write_lookup_to_rd);
        acc.fmadd(&w[6], &az.rd_write_eq_pc_plus_const_if_write_pc_to_rd);
        acc.fmadd(&w[7], &az.next_unexp_pc_eq_pc_plus_imm_if_should_branch);
        acc.fmadd(&w[8], &az.next_unexp_pc_update_otherwise);
        // Remaining weights (if any) multiply zero by construction
        acc.reduce()
    }

    /// Lagrange-folded Bz at r0 using base-domain weights (iterate up to group-1 length)
    #[inline]
    pub fn bz_at_r(&self, w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let bz = self.eval_bz();
        let mut acc: Acc7S<F> = Acc7S::new();
        acc.fmadd(&w[0], &bz.ram_addr_minus_rs1_plus_imm);
        acc.fmadd(&w[1], &bz.right_lookup_minus_add_result);
        acc.fmadd(&w[2], &bz.right_lookup_minus_sub_result);
        acc.fmadd(&w[3], &bz.right_lookup_minus_product);
        acc.fmadd(&w[4], &bz.right_lookup_minus_right_input);
        let rd_lookup_i128 = bz.rd_write_minus_lookup_output.to_i128();
        acc.fmadd(&w[5], &rd_lookup_i128);
        let rd_pc_i128 = bz.rd_write_minus_pc_plus_const;
        acc.fmadd(&w[6], &rd_pc_i128);
        acc.fmadd(&w[7], &bz.next_unexp_pc_minus_pc_plus_imm);
        acc.fmadd(&w[8], &bz.next_unexp_pc_minus_expected);
        acc.reduce()
    }

    #[cfg(test)]
    pub fn debug_assert_zero_when_guarded(&self) {
        let az = self.eval_az();
        let bz = self.eval_bz();
        debug_assert!((!az.ram_addr_eq_rs1_plus_imm_if_load_store) || bz.ram_addr_minus_rs1_plus_imm == 0i128);
        debug_assert!((!az.right_lookup_add) || bz.right_lookup_minus_add_result.is_zero());
        debug_assert!((!az.right_lookup_sub) || bz.right_lookup_minus_sub_result.is_zero());
        debug_assert!((!az.right_lookup_eq_product_if_mul) || bz.right_lookup_minus_product.is_zero());
        debug_assert!((!az.right_lookup_eq_right_input_otherwise) || bz.right_lookup_minus_right_input.is_zero());
        debug_assert!((az.rd_write_eq_lookup_if_write_lookup_to_rd == 0) || bz.rd_write_minus_lookup_output.to_i128() == 0);
        debug_assert!((az.rd_write_eq_pc_plus_const_if_write_pc_to_rd == 0) || bz.rd_write_minus_pc_plus_const == 0);
        debug_assert!((!az.next_unexp_pc_eq_pc_plus_imm_if_should_branch) || bz.next_unexp_pc_minus_pc_plus_imm == 0);
        debug_assert!((!az.next_unexp_pc_update_otherwise) || bz.next_unexp_pc_minus_expected == 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    /// Test that the constraint name enum matches the uniform R1CS order.
    #[test]
    fn constraint_enum_matches_uniform_r1cs_order() {
        let enum_order: Vec<ConstraintName> = ConstraintName::iter().collect();
        let array_order: Vec<ConstraintName> = UNIFORM_R1CS.iter().map(|nc| nc.name).collect();
        assert_eq!(array_order.len(), NUM_R1CS_CONSTRAINTS);
        assert_eq!(enum_order, array_order);
    }
}
