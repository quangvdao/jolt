//! Dense source, selector and fold tables for the five RV64I routers.

use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::views::{self, BaseWord, Selector};
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_arith::{BytecodeColumn, BytecodeRow};
use jolt_rv64i_verifier::ids::{
    CommittedPolynomial, DerivedId, OpeningId, RelationId, Router, RouterCycleDerived,
    RouterShortDerived, VirtualPolynomial,
};
use jolt_rv64i_verifier::points::{eq_index, equality_table, PointsError, WordLift};
use jolt_rv64i_verifier::public::routes::{projected_index, selector_slots, source_slots, ROUTERS};
use jolt_rv64i_verifier::stages::stage3a::RouterShort;
use jolt_rv64i_verifier::stages::stage3b::{
    RouterCycleBranch, RouterCycleCompare, RouterCycleMemory, RouterCycleShift, RouterCycleVariant,
};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;

fn fetched(witness: &Rv64iWitness, cycle: usize) -> Result<&BytecodeRow, Rv64iProverError> {
    let bits = witness.bits.get(cycle).ok_or(Rv64iProverError::RowCount {
        rows: witness.bits.len(),
    })?;
    let index = witness.layout.bytecode_index(bits);
    usize::try_from(index)
        .ok()
        .and_then(|index| witness.bytecode.rows().get(index))
        .filter(|row| row.variant.is_some())
        .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })
}

fn source_bank(
    witness: &Rv64iWitness,
    router: Router,
    cycle: usize,
    values: &mut [F128],
) -> Result<(), Rv64iProverError> {
    let bits = witness.bits.get(cycle).ok_or(Rv64iProverError::RowCount {
        rows: witness.bits.len(),
    })?;
    let words = witness.words.get(cycle).ok_or(Rv64iProverError::RowCount {
        rows: witness.words.len(),
    })?;
    let row = fetched(witness, cycle)?;
    let bank_words: &[u64] = match router {
        Router::Variant => &[
            words.rs1_value,
            words.rs2_value,
            words.rd_pre_value,
            row.imm,
            row.fall_through_pc,
            row.pc_plus_imm,
            row.pc,
            words.next_pc,
            witness.layout.inc(bits),
        ],
        Router::Shift => &[words.rs1_value],
        Router::Memory => &[words.ram_read_value, words.rs2_value],
        Router::Compare => &[words.rs1_value, words.rs2_value, row.imm],
        Router::Branch => &[row.fall_through_pc, row.pc_plus_imm],
    };
    values.fill(F128::zero());
    for (slot, word) in bank_words.iter().enumerate() {
        for bit in 0..64 {
            if let Some(value) = values.get_mut(64 * slot + bit) {
                *value = F128::from_u64((word >> bit) & 1);
            }
        }
    }
    match router {
        Router::Variant => {
            let g = witness
                .layout
                .ram_ra()
                .first()
                .map_or(64, |chunk| usize::from(chunk.start()));
            for column in g..witness.layout.used_columns() {
                if let Some(value) = values.get_mut(576 + column - g) {
                    *value = F128::from_u64(
                        bits.get(column / 64)
                            .map_or(0, |word| (word >> (column % 64)) & 1),
                    );
                }
            }
            if let Some(value) = values.get_mut(576 + witness.layout.used_columns() - g) {
                *value = F128::one();
            }
        }
        Router::Compare => {
            if let Some(value) = values.get_mut(192) {
                *value = F128::one();
            }
        }
        Router::Shift | Router::Memory | Router::Branch => {}
    }
    Ok(())
}

fn selector_bank(
    witness: &Rv64iWitness,
    router: Router,
    cycle: usize,
    values: &mut [F128],
) -> Result<(), Rv64iProverError> {
    let row = fetched(witness, cycle)?;
    let bits = witness.bits.get(cycle).ok_or(Rv64iProverError::RowCount {
        rows: witness.bits.len(),
    })?;
    let pos = usize::from(witness.layout.pos(bits));
    let selected = row.variant.and_then(|variant| match router {
        Router::Variant => Some(variant.index()),
        Router::Shift => variant.shift().map(|shift| pos + 64 * shift.kind as usize),
        Router::Memory => variant
            .access()
            .and_then(|access| access.kind)
            .map(|kind| (pos & 7) + 8 * kind as usize),
        Router::Compare => variant.key_kind().map(|kind| pos + 64 * kind as usize),
        Router::Branch => variant.branch().map(|_| 0),
    });
    values.fill(F128::zero());
    if let Some(value) = selected.and_then(|index| values.get_mut(index)) {
        *value = if router == Router::Branch {
            let column = witness.layout.should_branch();
            F128::from_u64(
                bits.get(column / 64)
                    .map_or(0, |word| (word >> (column % 64)) & 1),
            )
        } else {
            F128::one()
        };
    }
    Ok(())
}

/// Dense multilinear table in the router's source variables followed by its
/// selector variables, with the cycle variables summed at `r_1`.
pub fn fold(
    witness: &Rv64iWitness,
    router: Router,
    r_1: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let weights = equality_table(r_1)?;
    if weights.len() != witness.bits.len() {
        return Err(Rv64iProverError::RowCount {
            rows: witness.bits.len(),
        });
    }
    let source_size = 1 << source_slots(router).len();
    let mut values = vec![F128::zero(); source_size << selector_slots(router).len()];
    let mut source = vec![F128::zero(); source_size];
    let mut selectors = vec![F128::zero(); 1 << selector_slots(router).len()];
    for (cycle, weight) in weights.into_iter().enumerate() {
        source_bank(witness, router, cycle, &mut source)?;
        selector_bank(witness, router, cycle, &mut selectors)?;
        for (selector, select) in selectors.iter().copied().enumerate() {
            if select != F128::zero() {
                for (index, value) in source.iter().enumerate() {
                    if let Some(target) = values.get_mut(index + source_size * selector) {
                        *target += weight * *value * select;
                    }
                }
            }
        }
    }
    Ok(Polynomial::new(values))
}

#[derive(Default)]
pub struct RouterShortPrepare;
impl PrepareKernel<F128, RouterShort<F128>, Rv64iPlane> for RouterShortPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterShort<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterShort<F128>>>, KernelError<F128>>
    {
        let mut openings = BTreeMap::new();
        let mut derived = BTreeMap::new();
        let column_weights = equality_table(inputs.relation.w()).map_err(geometry_error)?;
        for router in ROUTERS {
            let active_fold =
                fold(witness, router, inputs.relation.r_1()).map_err(geometry_error)?;
            let source_size = 1 << source_slots(router).len();
            let full_fold = (0..1 << 17)
                .map(|index| {
                    let (source, selector) = projected_index(router, index);
                    active_fold
                        .evals()
                        .get(source + source_size * selector)
                        .copied()
                        .unwrap_or(F128::zero())
                })
                .collect();
            let _ = openings.insert(
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::RouterFold(router),
                    RelationId::RouterShort,
                ),
                Polynomial::new(full_fold),
            );
            let mut weight = vec![F128::zero(); 1 << 17];
            for entry in inputs.relation.routes().entries(router) {
                let index = source_slots(router)
                    .iter()
                    .enumerate()
                    .fold(0, |index, (bit, slot)| {
                        index | (((entry.source >> bit) & 1) << slot)
                    })
                    | selector_slots(router)
                        .iter()
                        .enumerate()
                        .fold(0, |index, (bit, slot)| {
                            index | (((entry.selector >> bit) & 1) << slot)
                        });
                if let Some(target) = weight.get_mut(index) {
                    *target += column_weights
                        .get(entry.column)
                        .copied()
                        .unwrap_or(F128::zero());
                }
            }
            let _ = derived.insert(
                DerivedId::RouterShort(RouterShortDerived::RouteWeight(router)),
                Polynomial::new(weight),
            );
        }
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}

fn geometry_error(error: impl Display) -> KernelError<F128> {
    KernelError::InvalidGeometry {
        reason: error.to_string(),
    }
}

fn variant_bits(
    witness: &Rv64iWitness,
    x: &[F128],
    lift: &WordLift<F128>,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let point = x.get(..10).ok_or(PointsError::Dimension {
        expected: 10,
        actual: x.len(),
    })?;
    let weights = equality_table(point)?;
    let g = witness
        .layout
        .ram_ra()
        .first()
        .map_or(64, |chunk| usize::from(chunk.start()));
    let inc_slot = eq_index(&point[6..10], 8)?;
    Ok(Polynomial::new(
        witness
            .bits
            .iter()
            .map(|row| {
                let inc = lift.evaluate(witness.layout.inc(row));
                inc_slot * inc
                    + (g..witness.layout.used_columns())
                        .map(|column| {
                            weights
                                .get(576 + column - g)
                                .copied()
                                .unwrap_or(F128::zero())
                                * F128::from_u64(
                                    row.get(column / 64)
                                        .map_or(0, |word| (word >> (column % 64)) & 1),
                                )
                        })
                        .sum::<F128>()
            })
            .collect(),
    ))
}

fn cycle_openings(
    witness: &Rv64iWitness,
    router: Router,
    x: &[F128],
) -> Result<BTreeMap<OpeningId, Polynomial<F128>>, Rv64iProverError> {
    if x.len() != 17 {
        return Err(PointsError::Dimension {
            expected: 17,
            actual: x.len(),
        }
        .into());
    }
    let bit = x.get(..6).ok_or(PointsError::Dimension {
        expected: 6,
        actual: x.len(),
    })?;
    let lift = WordLift::new(bit)?;
    let relation = match router {
        Router::Variant => RelationId::RouterCycleVariant,
        Router::Shift => RelationId::RouterCycleShift,
        Router::Memory => RelationId::RouterCycleMemory,
        Router::Compare => RelationId::RouterCycleCompare,
        Router::Branch => RelationId::RouterCycleBranch,
    };
    let mut tables = BTreeMap::new();
    let mut word = |id, table| {
        let _ = tables.insert(OpeningId::virtual_polynomial(id, relation), table);
    };
    match router {
        Router::Variant => {
            for (id, base) in [
                (VirtualPolynomial::Rs1Value, BaseWord::Rs1Value),
                (VirtualPolynomial::Rs2Value, BaseWord::Rs2Value),
                (VirtualPolynomial::RdPreValue, BaseWord::RdPreValue),
                (VirtualPolynomial::NextPC, BaseWord::NextPC),
            ] {
                word(id, views::base_word_with_lift(witness, base, &lift));
            }
            for (id, column) in [
                (VirtualPolynomial::Imm, BytecodeColumn::Imm),
                (
                    VirtualPolynomial::FallThroughPC,
                    BytecodeColumn::FallThroughPC,
                ),
                (VirtualPolynomial::PCPlusImm, BytecodeColumn::PCPlusImm),
                (VirtualPolynomial::PC, BytecodeColumn::PC),
            ] {
                word(id, views::bytecode_word_with_lift(witness, column, &lift)?);
            }
            word(
                VirtualPolynomial::Variant,
                views::selector(witness, Selector::Variant, &x[11..17])?,
            );
            let _ = tables.insert(
                OpeningId::committed(CommittedPolynomial::VariantBits, relation),
                variant_bits(witness, x, &lift)?,
            );
        }
        Router::Shift => {
            word(
                VirtualPolynomial::Rs1Value,
                views::base_word_with_lift(witness, BaseWord::Rs1Value, &lift),
            );
            word(
                VirtualPolynomial::ShiftKind,
                views::selector(witness, Selector::ShiftKind, &x[12..15])?,
            );
            for (digit, id) in [
                (0, CommittedPolynomial::PosRa0),
                (1, CommittedPolynomial::PosRa1),
            ] {
                let _ = tables.insert(
                    OpeningId::committed(id, relation),
                    views::chunk(
                        witness,
                        witness.layout.pos_ra()[digit],
                        &x[6 + 3 * digit..9 + 3 * digit],
                    )?,
                );
            }
        }
        Router::Memory => {
            word(
                VirtualPolynomial::RamReadValue,
                views::base_word_with_lift(witness, BaseWord::RamReadValue, &lift),
            );
            word(
                VirtualPolynomial::Rs2Value,
                views::base_word_with_lift(witness, BaseWord::Rs2Value, &lift),
            );
            word(
                VirtualPolynomial::AccessKind,
                views::selector(witness, Selector::AccessKind, &x[13..17])?,
            );
            let _ = tables.insert(
                OpeningId::committed(CommittedPolynomial::PosRa0, relation),
                views::chunk(witness, witness.layout.pos_ra()[0], &x[6..9])?,
            );
        }
        Router::Compare => {
            word(
                VirtualPolynomial::Rs1Value,
                views::base_word_with_lift(witness, BaseWord::Rs1Value, &lift),
            );
            word(
                VirtualPolynomial::Rs2Value,
                views::base_word_with_lift(witness, BaseWord::Rs2Value, &lift),
            );
            word(
                VirtualPolynomial::Imm,
                views::bytecode_word_with_lift(witness, BytecodeColumn::Imm, &lift)?,
            );
            word(
                VirtualPolynomial::KeyKind,
                views::selector(witness, Selector::KeyKind, &x[14..17])?,
            );
            for (digit, id) in [
                (0, CommittedPolynomial::PosRa0),
                (1, CommittedPolynomial::PosRa1),
            ] {
                let _ = tables.insert(
                    OpeningId::committed(id, relation),
                    views::chunk(
                        witness,
                        witness.layout.pos_ra()[digit],
                        &x[6 + 3 * digit..9 + 3 * digit],
                    )?,
                );
            }
        }
        Router::Branch => {
            word(
                VirtualPolynomial::FallThroughPC,
                views::bytecode_word_with_lift(witness, BytecodeColumn::FallThroughPC, &lift)?,
            );
            word(
                VirtualPolynomial::PCPlusImm,
                views::bytecode_word_with_lift(witness, BytecodeColumn::PCPlusImm, &lift)?,
            );
            word(VirtualPolynomial::Branch, views::branch(witness)?);
            let _ = tables.insert(
                OpeningId::committed(CommittedPolynomial::ShouldBranch, relation),
                views::bits_column(witness, witness.layout.should_branch()),
            );
        }
    }
    Ok(tables)
}

macro_rules! cycle_prepare {
    ($prepare:ident, $relation:ident, $router:ident) => {
        #[derive(Default)]
        pub struct $prepare;
        impl PrepareKernel<F128, $relation<F128>, Rv64iPlane> for $prepare {
            fn prepare(
                &self,
                _session: &mut ProofSession,
                witness: &Rv64iWitness,
                inputs: ProverInputs<'_, F128, $relation<F128>>,
            ) -> Result<Box<dyn SumcheckKernel<F128, Relation = $relation<F128>>>, KernelError<F128>> {
                let openings = cycle_openings(witness, Router::$router, inputs.relation.x()).map_err(geometry_error)?;
                let output_points = inputs.relation.derive_opening_points(&vec![F128::zero(); inputs.relation.r_1().len()], inputs.points).map_err(geometry_error)?;
                let ids: BTreeSet<_> = inputs.relation.symbolic().output_expression::<F128>().terms.iter().flat_map(|term| &term.factors).filter_map(|source| match source {
                    Source::Derived(id) => Some(*id),
                    Source::Opening(_) | Source::Challenge(_) => None,
                }).collect();
                let mut derived = BTreeMap::new();
                for id in ids {
                    let values = if id == DerivedId::RouterCycle(Router::$router, RouterCycleDerived::EqCycle) {
                        equality_table(inputs.relation.r_1()).map_err(geometry_error)?
                    } else {
                        vec![inputs.relation.derive_output_term(&id, inputs.points, &output_points, inputs.challenges).map_err(geometry_error)?; witness.bits.len()]
                    };
                    let _ = derived.insert(id, Polynomial::new(values));
                }
                Ok(Box::new(NaiveSumcheckProver::new(&inputs, openings, derived, BindingOrder::LowToHigh)?))
            }
        }
    };
}
cycle_prepare!(RouterCycleVariantPrepare, RouterCycleVariant, Variant);
cycle_prepare!(RouterCycleShiftPrepare, RouterCycleShift, Shift);
cycle_prepare!(RouterCycleMemoryPrepare, RouterCycleMemory, Memory);
cycle_prepare!(RouterCycleComparePrepare, RouterCycleCompare, Compare);
cycle_prepare!(RouterCycleBranchPrepare, RouterCycleBranch, Branch);
