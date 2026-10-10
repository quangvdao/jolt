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

use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::TransparentBits,
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
