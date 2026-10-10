//! Public router geometry and mixed-registry batch compatibility.
#![expect(
    clippy::unwrap_used,
    reason = "fixture failures fail their enclosing test"
)]

#[expect(
    dead_code,
    reason = "shared helpers serve the complete protocol corpus"
)]
mod support;

use jolt_rv64i_arith::Layout;
use jolt_rv64i_kernels::{
    router::shape::{selector_counts, RouterShape, RouterShapeRequest, SlotVariable, WordSlot},
    source::ValidatedTrace,
};
use jolt_rv64i_prover::optimized::{
    routers::router_shapes,
    source::{WitnessColumns, WitnessSource},
};
use jolt_rv64i_verifier::{
    ids::Router,
    public::routes::{idle_slots, selector_slots, source_slots, RouteTensors, ROUTERS},
};
use std::sync::Arc;
use support::PROGRAMS;

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
