//! Routers adapters for the packed RV64I kernels.

use crate::optimized::source::WitnessSource;
use crate::optimized::source::{SharedSource, WitnessColumns};
use crate::plane::{Rv64iPlane, Rv64iWitness};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_claims::NoChallenges;
use jolt_field::F128;
use jolt_kernels::mem::drop_in_background_thread;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_arith::Layout;
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::router::claims::claims_pass;
use jolt_rv64i_kernels::router::cycle::RouterCycleMember;
use jolt_rv64i_kernels::router::cycle::RoutersCycleCore;
use jolt_rv64i_kernels::router::fold::{fold_pass, FoldLayout};
use jolt_rv64i_kernels::router::lift::{source_lift, RetainedWordLifts};
use jolt_rv64i_kernels::router::shape::{
    BitEntry, RouteEntry, RouterError, RouterShape, RouterShapeRequest, SelectorFactor, WordSlot,
};
use jolt_rv64i_kernels::router::short::RouterShortCore;
use jolt_rv64i_kernels::source::ValidatedTrace;
use jolt_rv64i_verifier::ids::RouterCycleDerived;
use jolt_rv64i_verifier::ids::{DerivedId, Router, RouterShortDerived};
use jolt_rv64i_verifier::public::routes::{
    bank, selector_slots, source_slots, BankWord, Factor, RouteTensors, ROUTERS,
};
use jolt_rv64i_verifier::stages::stage3a::{
    RouterShort, RouterShortInputClaims, RouterShortOutputClaims,
};
use jolt_rv64i_verifier::stages::stage3b::{
    RouterCycleBranch, RouterCycleBranchOutputClaims, RouterCycleCompare,
    RouterCycleCompareOutputClaims, RouterCycleMemory, RouterCycleMemoryOutputClaims,
    RouterCycleShift, RouterCycleShiftOutputClaims, RouterCycleVariant,
    RouterCycleVariantInputClaims, RouterCycleVariantOutputClaims,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use std::fmt::Display;
use std::iter::Zip;
use std::marker::PhantomData;
use std::slice::Iter;
use std::sync::{Arc, Mutex};

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

#[derive(Default)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
struct CycleState {
    group: Option<Arc<Mutex<CycleGroup>>>,
}

#[cfg_attr(feature = "allocative", derive(Allocative))]
enum Extraction {
    Pending {
        #[cfg_attr(feature = "allocative", allocative(skip))]
        lifts: Box<RetainedWordLifts>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        plan: Arc<ScatterPlan<WitnessSource>>,
    },
    Complete {
        trace_words: Vec<(usize, F128)>,
        bytecode_words: Vec<F128>,
    },
}

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct CycleGroup {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    trace: Arc<ValidatedTrace<WitnessSource>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    members: Vec<Option<RouterCycleMember>>,
    r_1: Vec<F128>,
    x: Vec<F128>,
    extraction: Extraction,
    variant_terms: VariantTerms,
}

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct VariantWordTerm {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    word: BankWord,
    slot: usize,
    coefficient: F128,
}

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct VariantTerms {
    words: Vec<VariantWordTerm>,
    one: F128,
}
impl VariantTerms {
    fn new(shape: &RouterShape, layout: &Layout, x: &[F128]) -> Result<Self, RouterError> {
        if x.len() != shape.slots() {
            return Err(RouterError::PointLength {
                expected: shape.slots(),
                actual: x.len(),
            });
        }
        let word_point: Vec<_> = shape.word_slots().iter().map(|&slot| x[slot]).collect();
        let word_weights = eq_table(&word_point, None);
        let bit_weights = eq_table(&x[..6], None);
        let words = bank(Router::Variant, layout)
            .words
            .iter()
            .zip(&word_weights)
            .enumerate()
            .filter(|(_, (word, _))| **word != BankWord::Inc)
            .map(|(slot, (&word, &coefficient))| VariantWordTerm {
                word,
                slot,
                coefficient,
            })
            .collect();
        let mut one = F128::from_raw(0);
        for (slot, word) in shape.bank().iter().enumerate() {
            if let WordSlot::Bits(entries) = word {
                for (bit, entry) in entries.iter().enumerate() {
                    if *entry == BitEntry::One {
                        one += word_weights[slot] * bit_weights[bit];
                    }
                }
            }
        }
        Ok(Self { words, one })
    }
}

struct TakenMember {
    member: RouterCycleMember,
    group: Arc<Mutex<CycleGroup>>,
}
type FactorValues<'a> = Zip<Iter<'static, Factor>, Iter<'a, F128>>;

impl CycleGroup {
    fn new(
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        r_1: &[F128],
        x: &[F128],
    ) -> Result<Self, KernelError<F128>> {
        let columns = WitnessColumns::new(&witness.layout);
        let shapes = router_shapes(&columns, &witness.layout, None).map_err(geometry)?;
        let shared = session.state_or_insert_with(SharedSource::default);
        let trace = shared.prepare(witness, Some(RoutersCycleCore::columns(&shapes)))?;
        let plan = shared.plan()?;
        let variant_shape = ROUTERS
            .iter()
            .zip(&shapes)
            .find(|(router, _)| **router == Router::Variant)
            .map(|(_, shape)| shape)
            .ok_or_else(|| geometry("Variant shape is absent"))?;
        let variant_terms =
            VariantTerms::new(variant_shape, &witness.layout, x).map_err(geometry)?;
        let lifted = source_lift(&trace, &shapes, x).map_err(geometry)?;
        let core = RoutersCycleCore::new(
            shared.take_selector_group()?,
            &shapes,
            r_1,
            x,
            lifted.source_tables,
        )
        .map_err(geometry)?;
        Ok(Self {
            trace,
            members: core.members().into_iter().map(Some).collect(),
            r_1: r_1.to_vec(),
            x: x.to_vec(),
            extraction: Extraction::Pending {
                lifts: Box::new(lifted.lifts),
                plan,
            },
            variant_terms,
        })
    }

    fn take_member(
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        router: Router,
        r_1: &[F128],
        x: &[F128],
    ) -> Result<TakenMember, KernelError<F128>> {
        if session
            .state::<CycleState>()
            .and_then(|state| state.group.as_ref())
            .is_none()
        {
            let group = Arc::new(Mutex::new(Self::new(session, witness, r_1, x)?));
            session.state_or_insert_with(CycleState::default).group = Some(group);
        }
        let group = session
            .state::<CycleState>()
            .and_then(|state| state.group.as_ref())
            .cloned()
            .ok_or_else(|| geometry("cycle group is absent"))?;
        let member = {
            let mut state = group.lock().map_err(geometry)?;
            if state.r_1 != r_1 || state.x != x {
                return Err(geometry("router points differ from the cycle group"));
            }
            let index = ROUTERS
                .iter()
                .position(|name| *name == router)
                .ok_or_else(|| geometry("router is absent"))?;
            let member = state
                .members
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(|| geometry("router member was already taken"))?;
            member
        };
        Ok(TakenMember { member, group })
    }

    fn claims(&mut self, r_3: &[F128]) -> Result<(), SumcheckKernelError<F128>> {
        if let Extraction::Pending { lifts, plan } = &self.extraction {
            let words = [
                WitnessColumns::rs1_value(),
                WitnessColumns::rs2_value(),
                WitnessColumns::rd_pre_value(),
                WitnessColumns::ram_read_value(),
                WitnessColumns::next_pc(),
            ];
            let output =
                claims_pass(&self.trace, lifts, &words, plan, r_3).map_err(output_error)?;
            // Assignment drops the retained lifts and this group's plan Arc at extraction.
            self.extraction = Extraction::Complete {
                trace_words: words.into_iter().zip(output.trace_words).collect(),
                bytecode_words: output.bytecode_words,
            };
        }
        Ok(())
    }

    fn word(&self, word: BankWord) -> Result<F128, SumcheckKernelError<F128>> {
        let Extraction::Complete {
            trace_words,
            bytecode_words,
        } = &self.extraction
        else {
            return Err(output_error("claims are absent"));
        };
        let value = match word_slot(word) {
            WordSlot::Trace(index) => trace_words
                .iter()
                .find(|(word, _)| *word == index)
                .map(|(_, value)| value),
            WordSlot::Bytecode(index) => bytecode_words.get(index),
            WordSlot::Bits(_) | WordSlot::Zero => None,
        };
        value
            .copied()
            .ok_or_else(|| output_error("word claim is absent"))
    }
}

#[cfg_attr(feature = "allocative", derive(Allocative), allocative(bound = ""))]
struct CycleKernel<R> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    member: RouterCycleMember,
    group: Arc<Mutex<CycleGroup>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    factors: &'static [Factor],
    r_3: Vec<F128>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    relation: PhantomData<R>,
}

impl<R> ProveRounds<F128> for CycleKernel<R> {
    fn num_rounds(&self) -> usize {
        self.member.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let message = self.member.prove_round(bind, round, previous_claim)?;
        if let Some(bind) = bind {
            self.r_3.push(bind);
        }
        Ok(message)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.member.finish_rounds(bind)?;
        self.r_3.push(bind);
        Ok(())
    }
}

impl<R> CycleKernel<R> {
    fn factors(&self) -> Result<(F128, Vec<F128>), SumcheckKernelError<F128>> {
        let (source, values) = self.member.final_values().map_err(output_error)?;
        if values.len() != self.factors.len() {
            return Err(output_error("factor claims are absent"));
        }
        Ok((source, values))
    }
}

fn factor_value(
    values: &FactorValues<'_>,
    factor: Factor,
) -> Result<F128, SumcheckKernelError<F128>> {
    values
        .clone()
        .find(|(name, _)| **name == factor)
        .map(|(_, value)| *value)
        .ok_or_else(|| output_error("factor claim is absent"))
}

trait CycleAdapter: Send + Sync + 'static {
    type Relation: ConcreteSumcheck<F128>;
    const ROUTER: Router;
    const VALIDATES_DERIVED: bool = false;

    fn points(relation: &Self::Relation) -> (&[F128], &[F128]);
    fn output(
        state: &CycleGroup,
        source: F128,
        factors: &FactorValues<'_>,
    ) -> Result<SumcheckOutputClaims<F128, Self::Relation>, SumcheckKernelError<F128>>;
    fn validate(
        _state: &CycleGroup,
        _relation: &Self::Relation,
        _input_points: &SumcheckInputPoints<F128, Self::Relation>,
        _output_points: &SumcheckOutputPoints<F128, Self::Relation>,
        _challenges: &ConcreteSumcheckChallenges<F128, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        Ok(())
    }
}

fn prepare_cycle<A: CycleAdapter>(
    session: &mut ProofSession,
    witness: &Rv64iWitness,
    inputs: ProverInputs<'_, F128, A::Relation>,
) -> Result<Box<dyn SumcheckKernel<F128, Relation = A::Relation>>, KernelError<F128>> {
    let (r_1, x) = A::points(inputs.relation);
    let TakenMember { member, group } =
        CycleGroup::take_member(session, witness, A::ROUTER, r_1, x)?;
    Ok(Box::new(CycleKernel::<A> {
        member,
        group,
        factors: bank(A::ROUTER, &witness.layout).factors,
        r_3: Vec::with_capacity(r_1.len()),
        relation: PhantomData,
    }))
}

impl<A: CycleAdapter> SumcheckKernel<F128> for CycleKernel<A> {
    type Relation = A::Relation;

    fn output_claims(
        &mut self,
        _: &SumcheckInputClaims<F128, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F128, Self::Relation>, SumcheckKernelError<F128>> {
        let (source, values) = self.factors()?;
        let factors = self.factors.iter().zip(&values);
        let mut state = self.group.lock().map_err(output_error)?;
        state.claims(&self.r_3)?;
        A::output(&state, source, &factors)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F128, Self::Relation>,
        output_points: &SumcheckOutputPoints<F128, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F128, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        if !A::VALIDATES_DERIVED {
            return Ok(());
        }
        let state = self.group.lock().map_err(output_error)?;
        A::validate(&state, relation, input_points, output_points, challenges)
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        if let Some(state) = session.state::<CycleState>() {
            if state.group.is_some() {
                session.state_or_insert_with(CycleState::default).group = None;
                // plan() in the group's successful prepare pins a held session plan.
                let _ = session
                    .state_or_insert_with(SharedSource::default)
                    .release_plan();
            }
        }
        drop_in_background_thread(self);
    }
}

/// Shares the five-member `RoutersCycleCore`, retained lifts and plan.
/// Variant coefficients alone are checked against the relation's derived terms.
#[derive(Default)]
pub struct RouterCycleVariantPrepare;

impl PrepareKernel<F128, RouterCycleVariant<F128>, Rv64iPlane> for RouterCycleVariantPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterCycleVariant<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterCycleVariant<F128>>>, KernelError<F128>>
    {
        prepare_cycle::<Self>(session, witness, inputs)
    }
}

impl CycleAdapter for RouterCycleVariantPrepare {
    type Relation = RouterCycleVariant<F128>;
    const ROUTER: Router = Router::Variant;
    const VALIDATES_DERIVED: bool = true;

    fn points(relation: &Self::Relation) -> (&[F128], &[F128]) {
        (relation.r_1(), relation.x())
    }

    fn output(
        state: &CycleGroup,
        source: F128,
        factors: &FactorValues<'_>,
    ) -> Result<RouterCycleVariantOutputClaims<F128>, SumcheckKernelError<F128>> {
        let terms = &state.variant_terms;
        let rs1_value = state.word(BankWord::Rs1Value)?;
        let rs2_value = state.word(BankWord::Rs2Value)?;
        let rd_pre_value = state.word(BankWord::RdPreValue)?;
        let imm = state.word(BankWord::Imm)?;
        let fall_through_pc = state.word(BankWord::FallThroughPC)?;
        let pc_plus_imm = state.word(BankWord::PCPlusImm)?;
        let pc = state.word(BankWord::PC)?;
        let next_pc = state.word(BankWord::NextPC)?;
        let variant_bits = terms
            .words
            .iter()
            .try_fold(source + terms.one, |sum, term| {
                Ok::<_, SumcheckKernelError<F128>>(sum + state.word(term.word)? * term.coefficient)
            })?;
        Ok(RouterCycleVariantOutputClaims {
            rs1_value,
            rs2_value,
            rd_pre_value,
            imm,
            fall_through_pc,
            pc_plus_imm,
            pc,
            next_pc,
            variant_bits,
            variant: factor_value(factors, Factor::Variant)?,
        })
    }

    fn validate(
        state: &CycleGroup,
        relation: &Self::Relation,
        input_points: &RouterCycleVariantInputClaims<Vec<F128>>,
        output_points: &RouterCycleVariantOutputClaims<Vec<F128>>,
        challenges: &NoChallenges<F128>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        let terms = &state.variant_terms;
        for (term, got) in terms
            .words
            .iter()
            .map(|term| (RouterCycleDerived::WordSlot(term.slot), term.coefficient))
            .chain(std::iter::once((RouterCycleDerived::OneSlot, terms.one)))
        {
            let id = DerivedId::RouterCycle(Router::Variant, term);
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

/// Shares the five-member `RoutersCycleCore`, retained lifts and plan.
/// Variant coefficients alone are checked against the relation's derived terms.
#[derive(Default)]
pub struct RouterCycleShiftPrepare;

impl PrepareKernel<F128, RouterCycleShift<F128>, Rv64iPlane> for RouterCycleShiftPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterCycleShift<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterCycleShift<F128>>>, KernelError<F128>>
    {
        prepare_cycle::<Self>(session, witness, inputs)
    }
}

impl CycleAdapter for RouterCycleShiftPrepare {
    type Relation = RouterCycleShift<F128>;
    const ROUTER: Router = Router::Shift;

    fn points(relation: &Self::Relation) -> (&[F128], &[F128]) {
        (relation.r_1(), relation.x())
    }

    fn output(
        state: &CycleGroup,
        _source: F128,
        factors: &FactorValues<'_>,
    ) -> Result<RouterCycleShiftOutputClaims<F128>, SumcheckKernelError<F128>> {
        Ok(RouterCycleShiftOutputClaims {
            rs1_value: state.word(BankWord::Rs1Value)?,
            shift_kind: factor_value(factors, Factor::ShiftKind)?,
            pos_ra_0: factor_value(factors, Factor::Pos(0))?,
            pos_ra_1: factor_value(factors, Factor::Pos(1))?,
        })
    }
}

/// Shares the five-member `RoutersCycleCore`, retained lifts and plan.
/// Variant coefficients alone are checked against the relation's derived terms.
#[derive(Default)]
pub struct RouterCycleMemoryPrepare;

impl PrepareKernel<F128, RouterCycleMemory<F128>, Rv64iPlane> for RouterCycleMemoryPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterCycleMemory<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterCycleMemory<F128>>>, KernelError<F128>>
    {
        prepare_cycle::<Self>(session, witness, inputs)
    }
}

impl CycleAdapter for RouterCycleMemoryPrepare {
    type Relation = RouterCycleMemory<F128>;
    const ROUTER: Router = Router::Memory;

    fn points(relation: &Self::Relation) -> (&[F128], &[F128]) {
        (relation.r_1(), relation.x())
    }

    fn output(
        state: &CycleGroup,
        _source: F128,
        factors: &FactorValues<'_>,
    ) -> Result<RouterCycleMemoryOutputClaims<F128>, SumcheckKernelError<F128>> {
        Ok(RouterCycleMemoryOutputClaims {
            ram_read_value: state.word(BankWord::RamReadValue)?,
            rs2_value: state.word(BankWord::Rs2Value)?,
            access_kind: factor_value(factors, Factor::AccessKind)?,
            pos_ra_0: factor_value(factors, Factor::Pos(0))?,
        })
    }
}

/// Shares the five-member `RoutersCycleCore`, retained lifts and plan.
/// Variant coefficients alone are checked against the relation's derived terms.
#[derive(Default)]
pub struct RouterCycleComparePrepare;

impl PrepareKernel<F128, RouterCycleCompare<F128>, Rv64iPlane> for RouterCycleComparePrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterCycleCompare<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterCycleCompare<F128>>>, KernelError<F128>>
    {
        prepare_cycle::<Self>(session, witness, inputs)
    }
}

impl CycleAdapter for RouterCycleComparePrepare {
    type Relation = RouterCycleCompare<F128>;
    const ROUTER: Router = Router::Compare;

    fn points(relation: &Self::Relation) -> (&[F128], &[F128]) {
        (relation.r_1(), relation.x())
    }

    fn output(
        state: &CycleGroup,
        _source: F128,
        factors: &FactorValues<'_>,
    ) -> Result<RouterCycleCompareOutputClaims<F128>, SumcheckKernelError<F128>> {
        Ok(RouterCycleCompareOutputClaims {
            rs1_value: state.word(BankWord::Rs1Value)?,
            rs2_value: state.word(BankWord::Rs2Value)?,
            imm: state.word(BankWord::Imm)?,
            key_kind: factor_value(factors, Factor::KeyKind)?,
            pos_ra_0: factor_value(factors, Factor::Pos(0))?,
            pos_ra_1: factor_value(factors, Factor::Pos(1))?,
        })
    }
}

/// Shares the five-member `RoutersCycleCore`, retained lifts and plan.
/// Variant coefficients alone are checked against the relation's derived terms.
#[derive(Default)]
pub struct RouterCycleBranchPrepare;

impl PrepareKernel<F128, RouterCycleBranch<F128>, Rv64iPlane> for RouterCycleBranchPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RouterCycleBranch<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RouterCycleBranch<F128>>>, KernelError<F128>>
    {
        prepare_cycle::<Self>(session, witness, inputs)
    }
}

impl CycleAdapter for RouterCycleBranchPrepare {
    type Relation = RouterCycleBranch<F128>;
    const ROUTER: Router = Router::Branch;

    fn points(relation: &Self::Relation) -> (&[F128], &[F128]) {
        (relation.r_1(), relation.x())
    }

    fn output(
        state: &CycleGroup,
        _source: F128,
        factors: &FactorValues<'_>,
    ) -> Result<RouterCycleBranchOutputClaims<F128>, SumcheckKernelError<F128>> {
        Ok(RouterCycleBranchOutputClaims {
            fall_through_pc: state.word(BankWord::FallThroughPC)?,
            pc_plus_imm: state.word(BankWord::PCPlusImm)?,
            branch: factor_value(factors, Factor::Branch)?,
            should_branch: factor_value(factors, Factor::ShouldBranch)?,
        })
    }
}
