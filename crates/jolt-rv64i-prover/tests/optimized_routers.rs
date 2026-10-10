//! Public router geometry and mixed-registry batch compatibility.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture failures fail their enclosing test"
)]

#[expect(
    dead_code,
    reason = "shared helpers serve the complete protocol corpus"
)]
mod support;

use jolt_field::F128;
use jolt_kernels::ProofSession;
use jolt_rv64i_arith::Layout;
use jolt_rv64i_kernels::{
    router::shape::{selector_counts, RouterShape, RouterShapeRequest, SlotVariable, WordSlot},
    source::ValidatedTrace,
};
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    optimized::{
        routers::{
            router_shapes, RouterCycleBranchPrepare, RouterCycleComparePrepare,
            RouterCycleMemoryPrepare, RouterCycleShiftPrepare, RouterCycleVariantPrepare,
            RouterShortPrepare,
        },
        source::{WitnessColumns, WitnessSource},
    },
    stages::{stage3a, stage3b},
};
use jolt_rv64i_verifier::{
    ids::Router,
    proof::{BatchProof, RouterCycleValues, RouterFoldValues},
    public::routes::{idle_slots, selector_slots, source_slots, RouteTensors, ROUTERS},
    stages::{
        stage3a::{verify as short_verify, Output as ShortOutput},
        stage3b::Output as CycleOutput,
    },
    statement::CheckedInputs,
};
use jolt_sumcheck::ClearProof;
use std::sync::Arc;
use support::{ReferenceBatches, PROGRAMS};

#[test]
fn shapes_match_public_slots_routes_and_committed_bank_extent() {
    for (b, a) in [(4, 5), (10, 14), (20, 20), (3, 46)] {
        let layout = Layout::new(b, a, 0).unwrap();
        let columns = WitnessColumns::new(&layout);
        let routes = RouteTensors::new(&layout).unwrap();
        let shapes = router_shapes(&columns, &layout, Some(&routes)).unwrap();
        let without_routes = router_shapes(&columns, &layout, None).unwrap();
        assert_eq!(shapes.len(), ROUTERS.len());
        assert_eq!(without_routes.len(), ROUTERS.len());
        for ((router, shape), without) in ROUTERS.into_iter().zip(&shapes).zip(&without_routes) {
            let sources: Vec<_> = shape
                .slot_map()
                .iter()
                .filter_map(|&(slot, variable)| match variable {
                    SlotVariable::Bit(_) | SlotVariable::Word(_) => Some(slot),
                    SlotVariable::Selector { .. } => None,
                })
                .collect();
            assert_eq!(sources, source_slots(router), "{router:?} source slots");
            let selectors: Vec<_> = shape
                .factors()
                .iter()
                .flat_map(|factor| factor.slots.iter().copied())
                .collect();
            assert_eq!(
                selectors,
                selector_slots(router),
                "{router:?} selector slots"
            );
            assert_eq!(
                shape.idle_slots(),
                idle_slots(router),
                "{router:?} idle slots"
            );
            let found: Vec<_> = shape
                .route()
                .iter()
                .map(|entry| (entry.output, entry.source, entry.selector))
                .collect();
            let expected: Vec<_> = routes
                .entries(router)
                .iter()
                .map(|entry| (entry.column, entry.source, entry.selector))
                .collect();
            assert_eq!(found, expected, "{router:?} route tensor");
            assert_eq!(shape.bank(), without.bank(), "{router:?} bank");
            assert_eq!(shape.factors(), without.factors(), "{router:?} factors");
            assert_eq!(
                shape.word_slots(),
                without.word_slots(),
                "{router:?} word slots"
            );
            assert_eq!(
                shape.slot_map(),
                without.slot_map(),
                "{router:?} occupied slots"
            );
            assert_eq!(shape.slots(), without.slots());
            assert_eq!(shape.log_outputs(), without.log_outputs());
            assert!(without.route().is_empty(), "{router:?} without tensors");
            let rebuilt = RouterShape::new(RouterShapeRequest {
                slots: shape.slots(),
                bank: shape.bank().to_vec(),
                factors: shape.factors().to_vec(),
                word_slots: shape.word_slots().to_vec(),
                log_outputs: shape.log_outputs(),
                route: shape.route().to_vec(),
            })
            .unwrap();
            assert_eq!(rebuilt, *shape, "{router:?} checked shape");
            if router == Router::Variant && matches!((b, a), (20, 20) | (3, 46)) {
                let bits: Vec<_> = shape
                    .bank()
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, word)| matches!(word, WordSlot::Bits(_)).then_some(slot))
                    .collect();
                let expected = if a == 20 {
                    vec![9, 10]
                } else {
                    vec![9, 10, 11]
                };
                assert_eq!(bits, expected, "Variant Bits slots at ({b}, {a})");
            }
        }
    }
}

#[test]
fn variant_selector_counts_equal_replay_variant_counts() {
    for program in PROGRAMS {
        let (_, _, witness) = support::program_fixture(program, 6);
        let source = Arc::new(WitnessSource::new(&witness).unwrap());
        let shapes = router_shapes(source.columns(), &witness.layout, None).unwrap();
        let variant = ROUTERS
            .into_iter()
            .zip(&shapes)
            .find_map(|(router, shape)| (router == Router::Variant).then_some(shape))
            .unwrap();
        let trace = ValidatedTrace::new(source).unwrap();
        let found = selector_counts(&trace, variant).unwrap();
        let expected: Vec<_> = witness
            .variant_cycles
            .into_iter()
            .map(|count| count as usize)
            .collect();
        assert_eq!(found, expected, "{} Variant selectors", program.name());
    }
}

fn fold_values(values: &RouterFoldValues) -> [(&'static str, F128); 5] {
    [
        ("variant", values.variant),
        ("shift", values.shift),
        ("memory", values.memory),
        ("compare", values.compare),
        ("branch", values.branch),
    ]
}

fn cycle_values(values: &RouterCycleValues) -> [(&'static str, F128); 18] {
    [
        ("rs1_value", values.rs1_value),
        ("rs2_value", values.rs2_value),
        ("rd_pre_value", values.rd_pre_value),
        ("imm", values.imm),
        ("fall_through_pc", values.fall_through_pc),
        ("pc_plus_imm", values.pc_plus_imm),
        ("pc", values.pc),
        ("next_pc", values.next_pc),
        ("variant_bits", values.variant_bits),
        ("variant", values.variant),
        ("shift_kind", values.shift_kind),
        ("pos_ra_0", values.pos_ra_0),
        ("pos_ra_1", values.pos_ra_1),
        ("ram_read_value", values.ram_read_value),
        ("access_kind", values.access_kind),
        ("key_kind", values.key_kind),
        ("branch", values.branch),
        ("should_branch", values.should_branch),
    ]
}

fn assert_distinct(values: &[(&str, F128)], context: &str) {
    for (index, (name, value)) in values.iter().enumerate() {
        for (previous, previous_value) in &values[..index] {
            assert_ne!(
                value, previous_value,
                "{context}: {name} and {previous} coincide"
            );
        }
    }
}

fn assert_rounds<V>(found: &BatchProof<V>, expected: &BatchProof<V>, context: &str) {
    let ClearProof::Compressed(found) = found.rounds.as_clear().unwrap() else {
        panic!("{context}: proof is not compressed clear");
    };
    let ClearProof::Compressed(expected) = expected.rounds.as_clear().unwrap() else {
        panic!("{context}: reference is not compressed clear");
    };
    assert_eq!(
        found.round_polynomials.len(),
        expected.round_polynomials.len(),
        "{context}: first differing round {} (round count)",
        found
            .round_polynomials
            .len()
            .min(expected.round_polynomials.len()),
    );
    for (round, (found, expected)) in found
        .round_polynomials
        .iter()
        .zip(&expected.round_polynomials)
        .enumerate()
    {
        assert_eq!(found, expected, "{context}: first differing round {round}");
    }
}

fn assert_short(
    proof: &BatchProof<RouterFoldValues>,
    output: &ShortOutput,
    reference: &ReferenceBatches,
    context: &str,
) {
    assert_rounds(proof, &reference.stage3a.proof, context);
    let found_claims = fold_values(&short_verify::values(&output.claims));
    let expected_claims = fold_values(&short_verify::values(&reference.stage3a.output.claims));
    for ((((name, found), (_, expected)), (_, wire)), (_, reference_wire)) in found_claims
        .into_iter()
        .zip(expected_claims)
        .zip(fold_values(&proof.values))
        .zip(fold_values(&reference.stage3a.proof.values))
    {
        assert_eq!(
            found, expected,
            "{context}: first output router_short.{name}"
        );
        assert_eq!(wire, reference_wire, "{context}: first wire output {name}");
    }
    assert_eq!(
        output.x, reference.stage3a.output.x,
        "{context}: output point x"
    );
}

fn assert_cycle(
    proof: &BatchProof<RouterCycleValues>,
    output: &CycleOutput,
    reference: &ReferenceBatches,
    context: &str,
) {
    assert_rounds(proof, &reference.stage3b.proof, context);
    for ((name, found), (_, expected)) in cycle_values(&proof.values)
        .into_iter()
        .zip(cycle_values(&reference.stage3b.proof.values))
    {
        assert_eq!(found, expected, "{context}: first wire output {name}");
    }
    let expected = &reference.stage3b.output;
    macro_rules! member {
        ($member:ident, $($field:ident),+ $(,)?) => {
            $(
                assert_eq!(
                    output.claims.$member.$field,
                    expected.claims.$member.$field,
                    "{context}: first output {}.{}", stringify!($member), stringify!($field),
                );
                assert_eq!(
                    output.points.$member.$field,
                    expected.points.$member.$field,
                    "{context}: output point {}.{}", stringify!($member), stringify!($field),
                );
            )+
        };
    }
    member!(
        variant,
        rs1_value,
        rs2_value,
        rd_pre_value,
        imm,
        fall_through_pc,
        pc_plus_imm,
        pc,
        next_pc,
        variant_bits,
        variant
    );
    member!(shift, rs1_value, shift_kind, pos_ra_0, pos_ra_1);
    member!(memory, ram_read_value, rs2_value, access_kind, pos_ra_0);
    member!(compare, rs1_value, rs2_value, imm, key_kind, pos_ra_0, pos_ra_1);
    member!(branch, fall_through_pc, pc_plus_imm, branch, should_branch);
}

fn router_backend(mask: u8) -> Rv64iBackend {
    let mut backend = Rv64iBackend::reference();
    backend.stage3a.router_short = Box::new(RouterShortPrepare);
    if mask & 1 != 0 {
        backend.stage3b.variant = Box::new(RouterCycleVariantPrepare);
    }
    if mask & 2 != 0 {
        backend.stage3b.shift = Box::new(RouterCycleShiftPrepare);
    }
    if mask & 4 != 0 {
        backend.stage3b.memory = Box::new(RouterCycleMemoryPrepare);
    }
    if mask & 8 != 0 {
        backend.stage3b.compare = Box::new(RouterCycleComparePrepare);
    }
    if mask & 16 != 0 {
        backend.stage3b.branch = Box::new(RouterCycleBranchPrepare);
    }
    backend
}

#[test]
fn router_batches_match_reference_under_all_masks_and_both_session_states() {
    let (statement, preprocessing, witness) = support::separating_fixture();
    let reference = support::reference_batches(&statement, &preprocessing, &witness);
    assert_distinct(
        &fold_values(&reference.stage3a.proof.values),
        "batch 3a folds",
    );
    assert_distinct(
        &cycle_values(&reference.stage3b.proof.values),
        "batch 3b outputs",
    );
    let checked = CheckedInputs::of_statement(
        &preprocessing,
        &statement,
        witness.layout.log_K_ram() as u8,
        witness.final_pc,
    )
    .unwrap();
    let backend = router_backend(0);
    let mut transcript = reference.stage3a.transcript.fork();
    let (proof, output) = stage3a::prove(
        &checked,
        &witness,
        &backend.stage3a,
        &mut ProofSession::default(),
        &mut transcript,
        &reference.stage2.output,
    )
    .unwrap();
    assert_short(&proof, &output, &reference, "batch 3a optimised");
    for mask in 1..32 {
        let backend = router_backend(mask);
        for warm in [false, true] {
            let context = format!("batch 3b mask {mask:05b}, warm 3a {warm}");
            let mut session = ProofSession::default();
            let mut transcript = if warm {
                reference.stage3a.transcript.fork()
            } else {
                reference.stage3b.transcript.fork()
            };
            let warm_output = if warm {
                let (proof, output) = stage3a::prove(
                    &checked,
                    &witness,
                    &backend.stage3a,
                    &mut session,
                    &mut transcript,
                    &reference.stage2.output,
                )
                .unwrap();
                assert_short(&proof, &output, &reference, &context);
                Some(output)
            } else {
                None
            };
            let upstream = warm_output.as_ref().unwrap_or(&reference.stage3a.output);
            let (proof, output) = stage3b::prove(
                &checked,
                &witness,
                &backend.stage3b,
                &mut session,
                &mut transcript,
                &reference.stage1.output,
                upstream,
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            assert_cycle(&proof, &output, &reference, &context);
        }
    }
}
