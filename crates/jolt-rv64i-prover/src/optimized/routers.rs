//! Routers adapters for the packed RV64I kernels.

use crate::optimized::source::{SharedSource, WitnessColumns};
use crate::plane::{Rv64iPlane, Rv64iWitness};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_claims::NoChallenges;
use jolt_field::F128;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_arith::Layout;
use jolt_rv64i_kernels::router::cycle::RoutersCycleCore;
use jolt_rv64i_kernels::router::fold::{fold_pass, FoldLayout};
use jolt_rv64i_kernels::router::shape::{
    BitEntry, RouteEntry, RouterError, RouterShape, RouterShapeRequest, SelectorFactor, WordSlot,
};
use jolt_rv64i_kernels::router::short::RouterShortCore;
use jolt_rv64i_verifier::ids::{DerivedId, Router, RouterShortDerived};
use jolt_rv64i_verifier::public::routes::{
    bank, selector_slots, source_slots, BankWord, Factor, RouteTensors, ROUTERS,
};
use jolt_rv64i_verifier::stages::stage3a::{
    RouterShort, RouterShortInputClaims, RouterShortOutputClaims,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use std::fmt::Display;

fn geometry(error: impl Display) -> KernelError<F128> {
    KernelError::InvalidGeometry {
        reason: error.to_string(),
    }
}

fn word_slot(word: BankWord) -> WordSlot {
    match word {
        BankWord::Rs1Value => WordSlot::Trace(WitnessColumns::rs1_value()),
        BankWord::Rs2Value => WordSlot::Trace(WitnessColumns::rs2_value()),
        BankWord::RdPreValue => WordSlot::Trace(WitnessColumns::rd_pre_value()),
        BankWord::RamReadValue => WordSlot::Trace(WitnessColumns::ram_read_value()),
        BankWord::NextPC => WordSlot::Trace(WitnessColumns::next_pc()),
        BankWord::Inc => WordSlot::Trace(WitnessColumns::inc_word()),
        BankWord::Imm => WordSlot::Bytecode(WitnessColumns::imm()),
        BankWord::FallThroughPC => WordSlot::Bytecode(WitnessColumns::fall_through_pc()),
        BankWord::PCPlusImm => WordSlot::Bytecode(WitnessColumns::pc_plus_imm()),
        BankWord::PC => WordSlot::Bytecode(WitnessColumns::pc()),
        BankWord::One => WordSlot::Bits(vec![BitEntry::One]),
    }
}

fn factor_column(columns: &WitnessColumns, factor: Factor) -> Result<usize, RouterError> {
    Ok(match factor {
        Factor::Variant => columns.variant(),
        Factor::Pos(digit) => columns.pos(usize::from(digit)).ok_or(RouterError::Column {
            column: usize::from(digit),
            columns: 2,
        })?,
        Factor::ShiftKind => columns.shift_kind(),
        Factor::AccessKind => columns.access_kind(),
        Factor::KeyKind => columns.key_kind(),
        Factor::Branch => columns.branch(),
        Factor::ShouldBranch => columns.should_branch(),
    })
}

/// Converts the verifier-owned banks and variable maps into checked shapes in
/// `ROUTERS` order. With tensors, each support is copied once; without them no
/// route entries are walked. Passes additionally check columns against the source.
pub fn router_shapes(
    columns: &WitnessColumns,
    layout: &Layout,
    tensors: Option<&RouteTensors>,
) -> Result<Vec<RouterShape>, RouterError> {
    ROUTERS
        .into_iter()
        .map(|router| {
            let description = bank(router, layout);
            let mut words: Vec<_> = description.words.iter().copied().map(word_slot).collect();
            if let Some(committed) = description.committed {
                let mut entries = Vec::with_capacity(committed.len() + 1);
                for y in committed {
                    let (column, value) = columns.committed(y).ok_or(RouterError::Column {
                        column: y,
                        columns: layout.used_columns(),
                    })?;
                    entries.push(BitEntry::Indicator { column, value });
                }
                entries.push(BitEntry::One);
                words.extend(
                    entries
                        .chunks(64)
                        .map(|entries| WordSlot::Bits(entries.to_vec())),
                );
            }
            words.resize(words.len().next_power_of_two(), WordSlot::Zero);
            let selectors = selector_slots(router);
            let pos_width = |digit: u8| {
                layout
                    .pos_ra()
                    .get(usize::from(digit))
                    .map(|chunk| usize::from(chunk.bits()))
                    .ok_or(RouterError::Column {
                        column: usize::from(digit),
                        columns: layout.pos_ra().len(),
                    })
            };
            let mut pos_bits = 0;
            for factor in description.factors {
                if let Factor::Pos(digit) = *factor {
                    pos_bits += pos_width(digit)?;
                }
            }
            let kind_bits =
                selectors
                    .len()
                    .checked_sub(pos_bits)
                    .ok_or(RouterError::FactorWidth {
                        column: 0,
                        expected: pos_bits,
                        actual: selectors.len(),
                    })?;
            let mut start = 0;
            let mut factors = Vec::with_capacity(description.factors.len());
            for &factor in description.factors {
                let width = match factor {
                    Factor::Pos(digit) => pos_width(digit)?,
                    Factor::Branch | Factor::ShouldBranch => 0,
                    Factor::Variant | Factor::ShiftKind | Factor::AccessKind | Factor::KeyKind => {
                        kind_bits
                    }
                };
                let end = start + width;
                let slots = selectors.get(start..end).ok_or(RouterError::FactorWidth {
                    column: factor_column(columns, factor)?,
                    expected: end,
                    actual: selectors.len(),
                })?;
                factors.push(SelectorFactor {
                    column: factor_column(columns, factor)?,
                    slots: slots.to_vec(),
                });
                start = end;
            }
            let route = tensors.map_or_else(Vec::new, |tensors| {
                tensors
                    .entries(router)
                    .iter()
                    .map(|entry| RouteEntry {
                        output: entry.column,
                        source: entry.source,
                        selector: entry.selector,
                    })
                    .collect()
            });
            RouterShape::new(RouterShapeRequest {
                slots: 17,
                bank: words,
                factors,
                word_slots: source_slots(router).get(6..).unwrap_or(&[]).to_vec(),
                log_outputs: 10,
                route,
            })
        })
        .collect()
}

/// Runs `fold_pass` and `RouterShortCore`, sharing preparation and the plan with
/// batch 3b. Checks each final idle-weight scalar against `RouteWeight`.
#[derive(Default)]
pub struct RouterShortPrepare;

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct ShortKernel {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    core: RouterShortCore,
}

impl PrepareKernel<F128, RouterShort<F128>, Rv64iPlane> for RouterShortPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterShort<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterShort<F128>>>, KernelError<F128>>
    {
        let columns = WitnessColumns::new(&witness.layout);
        let shapes = router_shapes(&columns, &witness.layout, Some(inputs.relation.routes()))
            .map_err(geometry)?;
        let shared = session.state_or_insert_with(SharedSource::default);
        let trace = shared.prepare(witness, Some(RoutersCycleCore::columns(&shapes)))?;
        let plan = shared.plan()?;
        let mut values = vec![Vec::new(); shapes.len()];
        let variant = ROUTERS
            .iter()
            .position(|router| *router == Router::Variant)
            .ok_or_else(|| geometry("Variant shape is absent"))?;
        let counts = witness.variant_cycles.map(|count| count as usize);
        values[variant] =
            FoldLayout::byte_bucket_values(&counts, FoldLayout::DEFAULT_BYTE_BUCKET_LIMIT);
        let layout = FoldLayout::new(&trace, &shapes, &values).map_err(geometry)?;
        let folded = fold_pass(&trace, &shapes, inputs.relation.r_1(), &plan, &layout, &[])
            .map_err(geometry)?;
        drop(folded.ra_fold);
        let core =
            RouterShortCore::new(&shapes, inputs.relation.w(), folded.folds).map_err(geometry)?;
        Ok(Box::new(ShortKernel { core }))
    }
}

impl ProveRounds<F128> for ShortKernel {
    fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        self.core.prove_round(bind, round, previous_claim)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.core.finish_rounds(bind)
    }
}

fn output_error(_: impl Display) -> SumcheckKernelError<F128> {
    SumcheckKernelError::InvariantViolation {
        reason: "router output state is incomplete",
    }
}

impl SumcheckKernel<F128> for ShortKernel {
    type Relation = RouterShort<F128>;
    fn output_claims(
        &mut self,
        _: &RouterShortInputClaims<F128>,
    ) -> Result<RouterShortOutputClaims<F128>, SumcheckKernelError<F128>> {
        let pairs = self.core.final_values().map_err(output_error)?;
        let value = |router| {
            ROUTERS
                .iter()
                .zip(&pairs)
                .find(|(name, _)| **name == router)
                .map(|(_, pair)| pair.0)
                .ok_or_else(|| output_error("missing fold"))
        };
        Ok(RouterShortOutputClaims {
            variant: value(Router::Variant)?,
            shift: value(Router::Shift)?,
            memory: value(Router::Memory)?,
            compare: value(Router::Compare)?,
            branch: value(Router::Branch)?,
        })
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &RouterShortInputClaims<Vec<F128>>,
        output_points: &RouterShortOutputClaims<Vec<F128>>,
        challenges: &NoChallenges<F128>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        let pairs = self.core.final_values().map_err(output_error)?;
        for (router, (_, got)) in ROUTERS.into_iter().zip(pairs) {
            let id = DerivedId::RouterShort(RouterShortDerived::RouteWeight(router));
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift {
                    id: id.into(),
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}
