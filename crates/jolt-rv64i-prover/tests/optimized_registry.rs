//! Whole-proof byte compatibility and rejected inputs of the optimized registry.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture failures fail their enclosing test"
)]

#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;

use jolt_field::F128;
use jolt_prover::ProverError;
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::TransparentBits,
    error::Rv64iProverError,
    optimized::{
        outer::SpartanOuterF2Prepare,
        routers::{
            RouterCycleBranchPrepare, RouterCycleComparePrepare, RouterCycleMemoryPrepare,
            RouterCycleShiftPrepare, RouterCycleVariantPrepare, RouterShortPrepare,
        },
        tail::{BitsReductionPrepare, BytecodeReadCyclePrepare, RamRaProductPrepare},
    },
    prover::{prove, ProverPreprocessing},
};
use jolt_sumcheck::SumcheckError;
use support::{PROGRAMS, PROVING_FAILURES};

#[derive(Clone, Copy, Debug)]
enum Slot {
    Outer,
    Short,
    Variant,
    Shift,
    Memory,
    Compare,
    Branch,
    Bytecode,
    Ram,
    Bits,
}

const SLOTS: [Slot; 10] = [
    Slot::Outer,
    Slot::Short,
    Slot::Variant,
    Slot::Shift,
    Slot::Memory,
    Slot::Compare,
    Slot::Branch,
    Slot::Bytecode,
    Slot::Ram,
    Slot::Bits,
];

fn install(backend: &mut Rv64iBackend, slot: Slot) {
    match slot {
        Slot::Outer => backend.stage1.spartan_outer_f2 = Box::new(SpartanOuterF2Prepare),
        Slot::Short => backend.stage3a.router_short = Box::new(RouterShortPrepare),
        Slot::Variant => backend.stage3b.variant = Box::new(RouterCycleVariantPrepare),
        Slot::Shift => backend.stage3b.shift = Box::new(RouterCycleShiftPrepare),
        Slot::Memory => backend.stage3b.memory = Box::new(RouterCycleMemoryPrepare),
        Slot::Compare => backend.stage3b.compare = Box::new(RouterCycleComparePrepare),
        Slot::Branch => backend.stage3b.branch = Box::new(RouterCycleBranchPrepare),
        Slot::Bytecode => backend.stage6b.bytecode_read_cycle = Box::new(BytecodeReadCyclePrepare),
        Slot::Ram => backend.stage6b.ram_ra_product = Box::new(RamRaProductPrepare),
        Slot::Bits => backend.stage6b.bits_reduction = Box::new(BitsReductionPrepare),
    }
}

fn mixed(slots: &[Slot]) -> Rv64iBackend {
    let mut backend = Rv64iBackend::reference();
    for &slot in slots {
        install(&mut backend, slot);
    }
    backend
}

#[test]
fn mixed_registry_proofs_match_reference() {
    let (statement, verifier, witness) = support::counting_loop();
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let expected = prove::<TransparentBits>(
        &preprocessing,
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    )
    .unwrap()
    .to_bytes();
    let check = |name: &str, backend: Rv64iBackend| {
        let proof = prove::<TransparentBits>(&preprocessing, &statement, &witness, &backend)
            .unwrap_or_else(|error| panic!("registry {name}: {error}"));
        assert_eq!(proof.to_bytes(), expected, "registry {name}");
    };
    check("optimized", Rv64iBackend::optimized());
    for slot in SLOTS {
        check(&format!("{slot:?} alone"), mixed(&[slot]));
    }
    for (name, slots) in [
        ("five cycle routers", &SLOTS[2..7]),
        ("six routers", &SLOTS[1..7]),
        ("three tail members", &SLOTS[7..]),
        (
            "short and tail",
            &[Slot::Short, Slot::Bytecode, Slot::Ram, Slot::Bits][..],
        ),
    ] {
        check(name, mixed(slots));
    }
    check("all ten installed individually", mixed(&SLOTS));
}

#[test]
fn optimized_corpus_proofs_match_reference() {
    let reference = Rv64iBackend::reference();
    let optimized = Rv64iBackend::optimized();
    for program in PROGRAMS {
        for log_t in [6, 8, 10] {
            let (statement, verifier, witness) = support::program_fixture(program, log_t);
            let preprocessing = ProverPreprocessing {
                verifier,
                scheme: (),
            };
            let expected =
                prove::<TransparentBits>(&preprocessing, &statement, &witness, &reference)
                    .unwrap()
                    .to_bytes();
            let proof = prove::<TransparentBits>(&preprocessing, &statement, &witness, &optimized)
                .unwrap_or_else(|error| panic!("{} t={log_t}: {error}", program.name()));
            assert_eq!(proof.to_bytes(), expected, "{} t={log_t}", program.name());
        }
    }
}

fn failed_round(error: Rv64iProverError, expected_batch: &str) -> (usize, F128, F128) {
    let Rv64iProverError::Batch { batch, source } = error else {
        panic!("expected batch {expected_batch} rejection, found {error:?}");
    };
    assert_eq!(batch, expected_batch);
    let Rv64iProverError::Prover(ProverError::Sumcheck(SumcheckError::RoundCheckFailed {
        round,
        expected,
        actual,
    })) = *source
    else {
        panic!("expected RoundCheckFailed in batch {batch}, found {source:?}");
    };
    assert_eq!(round, 0);
    (round, expected, actual)
}

#[test]
fn optimized_rejects_the_same_public_statement_failures() {
    for case in PROVING_FAILURES {
        let fixture = support::proving_failure_fixture(case);
        let preprocessing = ProverPreprocessing {
            verifier: fixture.verifier,
            scheme: (),
        };
        let failure = |backend: Rv64iBackend| {
            let error = prove::<TransparentBits>(
                &preprocessing,
                &fixture.changed,
                &fixture.witness,
                &backend,
            )
            .err()
            .unwrap_or_else(|| panic!("{case:?}: expected proving failure"));
            failed_round(error, fixture.expected_batch)
        };
        assert_eq!(
            failure(Rv64iBackend::optimized()),
            failure(Rv64iBackend::reference()),
            "{case:?}"
        );
    }
}
