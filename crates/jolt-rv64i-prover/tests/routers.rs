//! Independent machine and cube oracles for the two binary-field RV64I router batches.
#![expect(clippy::unwrap_used, reason = "invalid test fixtures fail the test")]

mod support;

use blake2::{digest::consts::U32, Blake2b, Digest};
use common::constants::RAM_START_ADDRESS;
use jolt_crypto::NoCommitment;
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::ProofSession;
use jolt_poly::Polynomial;
use jolt_prover::driver::StageProver;
use jolt_rv64i_arith::decode::RdWriteSource;
use jolt_rv64i_arith::{BitsRow, BytecodeRow, Layout, RowSystem, Variant, WitnessRow};
use jolt_rv64i_prover::commitment::transparent::TransparentBits;
use jolt_rv64i_prover::plane::{CycleWords, Rv64iWitness};
use jolt_rv64i_prover::stages::{
    stage3a::{Stage3aKernels, Stage3aSumchecks},
    stage3b::{Stage3bKernels, Stage3bSumchecks},
};
use jolt_rv64i_verifier::claims::{
    router_cycle::{
        RouterCycleBranchInputClaims, RouterCycleCompareInputClaims, RouterCycleMemoryInputClaims,
        RouterCycleShiftInputClaims, RouterCycleVariantInputClaims,
    },
    router_short::RouterShortInputClaims,
};
use jolt_rv64i_verifier::ids::Router;
use jolt_rv64i_verifier::points::{eq_index, to_high_to_low};
use jolt_rv64i_verifier::proof::BatchProof;
use jolt_rv64i_verifier::public::routes::{RouteTensors, ROUTERS};
use jolt_rv64i_verifier::stages::stage3b::verify::{
    expand as expand_cycle, values as cycle_values,
};
use jolt_rv64i_verifier::stages::{stage3a, stage3b};
use jolt_rv64i_verifier::stages::{
    stage3a::{Stage3aInputClaims, Stage3aInputPoints, Stage3aSumchecks as VerifierStage3a},
    stage3b::{Stage3bInputClaims, Stage3bSumchecks as VerifierStage3b},
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeSet;
use std::sync::Arc;

type BinaryTranscript = Blake2bTranscript<F128>;

fn bit(bits: &BitsRow, column: usize) -> bool {
    (bits[column / 64] >> (column % 64)) & 1 != 0
}
fn scalar(value: bool) -> F128 {
    F128::from_u64(u64::from(value))
}
fn point(rng: &mut StdRng, size: usize) -> Vec<F128> {
    (0..size).map(|_| F128::from_raw(rng.gen())).collect()
}
fn virtual_columns() -> impl Iterator<Item = usize> {
    [1, 2, 3].into_iter().chain(16..27).chain(64..768)
}
fn fetched(witness: &Rv64iWitness, cycle: usize) -> &BytecodeRow {
    &witness.bytecode.rows()[witness.layout.bytecode_index(&witness.bits[cycle]) as usize]
}

// These slot lists are the test's literal cube domains of §8.4.
fn domains(router: Router) -> (&'static [usize], &'static [usize], &'static [usize]) {
    match router {
        Router::Variant => (
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[11, 12, 13, 14, 15, 16],
            &[10],
        ),
        Router::Shift => (
            &[0, 1, 2, 3, 4, 5],
            &[6, 7, 8, 9, 10, 11, 12, 13, 14],
            &[15, 16],
        ),
        Router::Memory => (
            &[0, 1, 2, 3, 4, 5, 12],
            &[6, 7, 8, 13, 14, 15, 16],
            &[9, 10, 11],
        ),
        Router::Compare => (
            &[0, 1, 2, 3, 4, 5, 12, 13],
            &[6, 7, 8, 9, 10, 11, 14, 15, 16],
            &[],
        ),
        Router::Branch => (
            &[0, 1, 2, 3, 4, 5, 12],
            &[],
            &[6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
        ),
    }
}
fn source(
    router: Router,
    layout: &Layout,
    row: &BytecodeRow,
    bits: &BitsRow,
    words: &CycleWords,
    index: usize,
) -> bool {
    let word = match router {
        Router::Variant => {
            if index >= 576 {
                let g = usize::from(layout.ram_ra()[0].start());
                let y = g + index - 576;
                return if y < layout.used_columns() {
                    bit(bits, y)
                } else {
                    y == layout.used_columns()
                };
            }
            match index / 64 {
                0 => words.rs1_value,
                1 => words.rs2_value,
                2 => words.rd_pre_value,
                3 => row.imm,
                4 => row.fall_through_pc,
                5 => row.pc_plus_imm,
                6 => row.pc,
                7 => words.next_pc,
                8 => layout.inc(bits),
                _ => 0,
            }
        }
        Router::Shift => words.rs1_value,
        Router::Memory => {
            if index < 64 {
                words.ram_read_value
            } else {
                words.rs2_value
            }
        }
        Router::Compare => match index / 64 {
            0 => words.rs1_value,
            1 => words.rs2_value,
            2 => row.imm,
            3 => 1,
            _ => 0,
        },
        Router::Branch => {
            if index < 64 {
                row.fall_through_pc
            } else {
                row.pc_plus_imm
            }
        }
    };
    (word >> (index % 64)) & 1 != 0
}
fn selector(
    router: Router,
    layout: &Layout,
    variant: Variant,
    bits: &BitsRow,
    index: usize,
) -> bool {
    let low = layout.pos_ra()[0].full(bits);
    let high = layout.pos_ra()[1].full(bits);
    match router {
        Router::Variant => index == variant.index(),
        Router::Shift => {
            variant
                .shift()
                .is_some_and(|s| index >> 6 == s.kind.index())
                && (low >> (index & 7)) & 1 != 0
                && (high >> ((index >> 3) & 7)) & 1 != 0
        }
        Router::Memory => {
            variant
                .access()
                .and_then(|a| a.kind)
                .is_some_and(|k| index >> 3 == k.index())
                && (low >> (index & 7)) & 1 != 0
        }
        Router::Compare => {
            variant.key_kind().is_some_and(|k| index >> 6 == k.index())
                && (low >> (index & 7)) & 1 != 0
                && (high >> ((index >> 3) & 7)) & 1 != 0
        }
        Router::Branch => variant.branch().is_some() && bit(bits, layout.should_branch()),
    }
}
fn source_at(witness: &Rv64iWitness, router: Router, x: &[F128], cycle: usize) -> F128 {
    let (slots, _, _) = domains(router);
    let p: Vec<_> = slots.iter().map(|&k| x[k]).collect();
    (0..1 << slots.len())
        .filter(|&s| {
            source(
                router,
                &witness.layout,
                fetched(witness, cycle),
                &witness.bits[cycle],
                &witness.words[cycle],
                s,
            )
        })
        .map(|s| eq_index(&p, s).unwrap())
        .sum()
}
fn select_at(witness: &Rv64iWitness, router: Router, x: &[F128], cycle: usize) -> F128 {
    let (_, slots, _) = domains(router);
    let p: Vec<_> = slots.iter().map(|&k| x[k]).collect();
    let variant = fetched(witness, cycle).variant.unwrap();
    (0..1 << slots.len())
        .filter(|&h| selector(router, &witness.layout, variant, &witness.bits[cycle], h))
        .map(|h| eq_index(&p, h).unwrap())
        .sum()
}
fn evaluate(values: Vec<F128>, point: &[F128]) -> F128 {
    Polynomial::new(values).evaluate(&to_high_to_low(point))
}
fn weight_at(routes: &RouteTensors, router: Router, w: &[F128], x: &[F128]) -> F128 {
    let (src, sel, idle) = domains(router);
    let p: Vec<_> = src.iter().map(|&k| x[k]).collect();
    let q: Vec<_> = sel.iter().map(|&k| x[k]).collect();
    let columns: Vec<_> = (0..1024).map(|c| eq_index(w, c).unwrap()).collect();
    let sources: Vec<_> = (0..1 << p.len())
        .map(|s| eq_index(&p, s).unwrap())
        .collect();
    let selectors: Vec<_> = (0..1 << q.len())
        .map(|h| eq_index(&q, h).unwrap())
        .collect();
    let sum: F128 = routes
        .entries(router)
        .iter()
        .map(|e| columns[e.column] * sources[e.source] * selectors[e.selector])
        .sum();
    idle.iter().fold(sum, |v, &k| v * (F128::one() + x[k]))
}

#[test]
fn sparse_route_tensors_match_ten_thousand_cycles_at_two_layouts() {
    let mut rng = StdRng::seed_from_u64(0x726f_7574_6573);
    for (b, a) in [(4, 5), (20, 20)] {
        let layout = Layout::new(b, a, 0).unwrap();
        let routes = RouteTensors::new(&layout).unwrap();
        let buckets: Vec<Vec<Vec<_>>> = ROUTERS
            .into_iter()
            .map(|r| {
                let mut buckets = vec![Vec::new(); 1 << domains(r).1.len()];
                for entry in routes.entries(r) {
                    buckets[entry.selector].push(entry);
                }
                buckets
            })
            .collect();
        for j in 0..10_000 {
            let variant = Variant::ALL[j % 58];
            let row = BytecodeRow {
                variant: Some(variant),
                pc: rng.gen(),
                imm: rng.gen(),
                fall_through_pc: rng.gen(),
                pc_plus_imm: rng.gen(),
                ..BytecodeRow::default()
            };
            let mut bits = rng.gen::<BitsRow>();
            layout
                .write_bytecode_index(&mut bits, rng.gen_range(0..1_u64 << b))
                .unwrap();
            layout
                .write_ram_index(&mut bits, rng.gen_range(0..1_u64 << a))
                .unwrap();
            layout.write_pos(&mut bits, rng.gen_range(0..64)).unwrap();
            let words = CycleWords {
                rs1_value: rng.gen(),
                rs2_value: rng.gen(),
                rd_pre_value: rng.gen(),
                ram_read_value: rng.gen(),
                next_pc: rng.gen(),
            };
            let base = words.base_words(variant.is_store(), layout.inc(&bits));
            let expected = WitnessRow::compute(&layout, &row, &base, &bits);
            let mut actual = [false; 1024];
            for (r, bucket) in ROUTERS.into_iter().zip(&buckets) {
                for (_, entries) in bucket
                    .iter()
                    .enumerate()
                    .filter(|(h, _)| selector(r, &layout, variant, &bits, *h))
                {
                    for entry in entries {
                        actual[entry.column] ^=
                            source(r, &layout, &row, &bits, &words, entry.source);
                    }
                }
            }
            for c in virtual_columns() {
                assert_eq!(
                    actual[c],
                    expected.bit(c).unwrap() ^ (c == 16),
                    "layout {b},{a}, variant {variant:?}, column {c}, cycle {j}"
                );
            }
            let expanded = variant.rd_write_sources().fold(0, |value, term| {
                value
                    ^ match term {
                        RdWriteSource::RdPreValue => words.rd_pre_value,
                        RdWriteSource::Inc => layout.inc(&bits),
                    }
            });
            assert_eq!(expanded, base.rd_write_value);
        }
    }
}

#[test]
fn route_tensor_encoding_and_nonzero_count_are_frozen() {
    for (b, a, count, frozen_digest) in [
        (
            20,
            20,
            75371,
            [
                98, 133, 46, 132, 126, 82, 188, 51, 197, 238, 139, 77, 163, 7, 1, 156, 19, 28, 221,
                37, 86, 131, 75, 42, 39, 221, 14, 65, 205, 144, 82, 22,
            ],
        ),
        (
            4,
            5,
            69_191,
            [
                246, 16, 181, 100, 24, 184, 185, 3, 34, 250, 177, 82, 54, 155, 132, 34, 156, 251,
                40, 224, 233, 218, 206, 159, 12, 81, 94, 123, 127, 16, 185, 237,
            ],
        ),
    ] {
        let layout = Layout::new(b, a, 0).unwrap();
        let routes = RouteTensors::new(&layout).unwrap();
        let mut digest = Blake2b::<U32>::new();
        for r in ROUTERS {
            digest.update([r as u8]);
            digest.update((routes.entries(r).len() as u64).to_le_bytes());
            for entry in routes.entries(r) {
                for value in [entry.column, entry.source, entry.selector] {
                    digest.update((value as u64).to_le_bytes());
                }
            }
        }
        let actual: [u8; 32] = digest.finalize().into();
        assert_eq!(
            (routes.nonzero_entries(), actual),
            (count, frozen_digest),
            "layout {b},{a}"
        );
    }
}

fn word_at(word: u64, p: &[F128]) -> F128 {
    (0..64)
        .filter(|i| (word >> i) & 1 != 0)
        .map(|i| eq_index(p, i).unwrap())
        .sum()
}
fn position_at(witness: &Rv64iWitness, cycle: usize, digit: usize, x: &[F128]) -> F128 {
    let full = witness.layout.pos_ra()[digit].full(&witness.bits[cycle]);
    (0..8)
        .filter(|k| (full >> k) & 1 != 0)
        .map(|k| eq_index(&x[6 + 3 * digit..9 + 3 * digit], k).unwrap())
        .sum()
}
fn kind_at(witness: &Rv64iWitness, cycle: usize, router: Router, x: &[F128]) -> F128 {
    let v = fetched(witness, cycle).variant.unwrap();
    let (index, p) = match router {
        Router::Variant => (Some(v.index()), &x[11..17]),
        Router::Shift => (v.shift().map(|s| s.kind.index()), &x[12..15]),
        Router::Memory => (
            v.access().and_then(|a| a.kind).map(|k| k.index()),
            &x[13..17],
        ),
        Router::Compare => (v.key_kind().map(|k| k.index()), &x[14..17]),
        Router::Branch => return scalar(v.branch().is_some()),
    };
    index.map_or(F128::zero(), |i| eq_index(p, i).unwrap())
}
fn terminal_selector(witness: &Rv64iWitness, router: Router, x: &[F128], r3: &[F128]) -> F128 {
    let n = witness.bits.len();
    let kind = evaluate((0..n).map(|j| kind_at(witness, j, router, x)).collect(), r3);
    let low = || evaluate((0..n).map(|j| position_at(witness, j, 0, x)).collect(), r3);
    let high = || evaluate((0..n).map(|j| position_at(witness, j, 1, x)).collect(), r3);
    match router {
        Router::Variant => kind,
        Router::Shift | Router::Compare => kind * low() * high(),
        Router::Memory => kind * low(),
        Router::Branch => {
            kind * evaluate(
                witness
                    .bits
                    .iter()
                    .map(|bits| scalar(bit(bits, witness.layout.should_branch())))
                    .collect(),
                r3,
            )
        }
    }
}
fn routed_claim(witness: &Rv64iWitness, w: &[F128], r1: &[F128]) -> F128 {
    let weights: Vec<_> = (0..1024).map(|c| eq_index(w, c).unwrap()).collect();
    witness
        .bits
        .iter()
        .enumerate()
        .map(|(j, bits)| {
            let row = fetched(witness, j);
            let base = witness.words[j]
                .base_words(row.variant.unwrap().is_store(), witness.layout.inc(bits));
            let z = WitnessRow::compute(&witness.layout, row, &base, bits);
            let value: F128 = virtual_columns()
                .filter(|&c| z.bit(c).unwrap() ^ (c == 16))
                .map(|c| weights[c])
                .sum();
            eq_index(r1, j).unwrap() * value
        })
        .sum()
}
fn short_cube_sum(witness: &Rv64iWitness, routes: &RouteTensors, w: &[F128], r1: &[F128]) -> F128 {
    let column_weights: Vec<_> = (0..1024).map(|c| eq_index(w, c).unwrap()).collect();
    let cycle_weights: Vec<_> = (0..witness.bits.len())
        .map(|j| eq_index(r1, j).unwrap())
        .collect();
    let mut total = F128::zero();
    for router in ROUTERS {
        let (src, sel, idle) = domains(router);
        let mut weights = vec![F128::zero(); 1 << 17];
        for entry in routes.entries(router) {
            let index = src
                .iter()
                .enumerate()
                .fold(0, |u, (k, slot)| u | (((entry.source >> k) & 1) << slot))
                | sel
                    .iter()
                    .enumerate()
                    .fold(0, |u, (k, slot)| u | (((entry.selector >> k) & 1) << slot));
            weights[index] += column_weights[entry.column];
        }
        let mut folds = vec![F128::zero(); 1 << (src.len() + sel.len())];
        for (j, &cycle_weight) in cycle_weights.iter().enumerate() {
            let row = fetched(witness, j);
            let sources: Vec<_> = (0..1 << src.len())
                .filter(|&s| {
                    source(
                        router,
                        &witness.layout,
                        row,
                        &witness.bits[j],
                        &witness.words[j],
                        s,
                    )
                })
                .collect();
            for h in (0..1 << sel.len()).filter(|&h| {
                selector(
                    router,
                    &witness.layout,
                    row.variant.unwrap(),
                    &witness.bits[j],
                    h,
                )
            }) {
                for &s in &sources {
                    folds[s | (h << src.len())] += cycle_weight;
                }
            }
        }
        for (u, weight) in weights.into_iter().enumerate() {
            if !weight.is_zero() && idle.iter().all(|slot| (u >> slot) & 1 == 0) {
                let s = src
                    .iter()
                    .enumerate()
                    .fold(0, |v, (k, slot)| v | (((u >> slot) & 1) << k));
                let h = sel
                    .iter()
                    .enumerate()
                    .fold(0, |v, (k, slot)| v | (((u >> slot) & 1) << k));
                total += weight * folds[s | (h << src.len())];
            }
        }
    }
    total
}

fn prove_router_batches(
    witness: &Rv64iWitness,
    checked: Option<&CheckedInputs<'_, TransparentBits>>,
) {
    let mut rng = StdRng::seed_from_u64(0x033a_033b);
    let w = point(&mut rng, 10);
    let r1 = point(&mut rng, witness.bits.len().ilog2() as usize);
    let routes = Arc::new(RouteTensors::new(&witness.layout).unwrap());
    let short =
        Stage3aSumchecks(VerifierStage3a::new(w.clone(), r1.clone(), Arc::clone(&routes)).unwrap());
    let routed = routed_claim(witness, &w, &r1);
    assert_eq!(routed, short_cube_sum(witness, &routes, &w, &r1));
    let inputs = Stage3aInputClaims {
        router_short: RouterShortInputClaims {
            witness_routed: routed,
        },
    };
    let input_points = Stage3aInputPoints {
        router_short: short.router_short.input_points(),
    };
    let mut prover = BinaryTranscript::new(b"rv64i-router-batches");
    let mut verifier = BinaryTranscript::new(b"rv64i-router-batches");
    let mut stage_transcript = BinaryTranscript::new(b"rv64i-router-batches");
    let challenges = short.draw_challenges(&mut prover).unwrap();
    assert_eq!(
        short
            .router_short
            .input_claim(&inputs.router_short, &challenges.router_short)
            .unwrap(),
        routed
    );
    let proof = short
        .prove(
            &Stage3aKernels::default(),
            &mut ProofSession::default(),
            &mut SequentialRounds,
            witness,
            &inputs,
            &input_points,
            &challenges,
            ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut prover,
        )
        .unwrap();
    let twin = short.draw_challenges(&mut verifier).unwrap();
    let outputs = short
        .verify_clear(
            &inputs,
            &input_points,
            &twin,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut verifier,
            0,
        )
        .unwrap();
    short.append_output_claims(&mut verifier, &proof.output_claims);
    assert_eq!(prover.state(), verifier.state());
    assert_eq!(outputs, proof.output_points);
    let stage3a_output = checked.map(|checked| {
        let wire = BatchProof {
            rounds: proof.recorded.proof.clone(),
            values: stage3a::verify::values(&proof.output_claims),
        };
        let output =
            stage3a::verify::verify(checked, &wire, &mut stage_transcript, &w, &r1, routed)
                .unwrap();
        assert_eq!(output.points, outputs);
        assert_eq!(stage_transcript.state(), prover.state());
        output
    });
    let mut x = vec![F128::zero(); 17];
    let p = &proof.output_points.router_short;
    let values = &proof.output_claims.router_short;
    for (router, restricted) in ROUTERS
        .into_iter()
        .zip([&p.variant, &p.shift, &p.memory, &p.compare, &p.branch])
    {
        let (src, sel, _) = domains(router);
        for (&slot, &value) in src.iter().chain(sel).zip(restricted) {
            x[slot] = value;
        }
    }
    // Variant omits slot 10; Compare includes every slot and fixes it here.
    for (&slot, &value) in domains(Router::Compare)
        .0
        .iter()
        .chain(domains(Router::Compare).1)
        .zip(&p.compare)
    {
        x[slot] = value;
    }
    for (router, restricted) in ROUTERS
        .into_iter()
        .zip([&p.variant, &p.shift, &p.memory, &p.compare, &p.branch])
    {
        let (src, sel, _) = domains(router);
        let expected: Vec<_> = src.iter().chain(sel).map(|&slot| x[slot]).collect();
        assert_eq!(restricted, &expected, "short {router:?} opening point");
    }
    let folds = [
        values.variant,
        values.shift,
        values.memory,
        values.compare,
        values.branch,
    ];
    if (0..witness.bits.len()).any(|j| fetched(witness, j).variant.unwrap().shift().is_some()) {
        for (router, fold) in ROUTERS.into_iter().zip(folds) {
            assert!(!fold.is_zero(), "nonzero synthetic {router:?} fold");
        }
    }
    for (router, &fold) in ROUTERS.iter().zip(&folds) {
        let direct: F128 = (0..witness.bits.len())
            .map(|j| {
                eq_index(&r1, j).unwrap()
                    * source_at(witness, *router, &x, j)
                    * select_at(witness, *router, &x, j)
            })
            .sum();
        assert_eq!(fold, direct);
    }
    let terminal: F128 = ROUTERS
        .into_iter()
        .zip(folds)
        .map(|(router, fold)| weight_at(&routes, router, &w, &x) * fold)
        .sum();
    assert_eq!(
        short
            .router_short
            .expected_output(
                &input_points.router_short,
                values,
                p,
                &challenges.router_short
            )
            .unwrap(),
        terminal
    );

    let short_proof = proof;
    let short_inputs = inputs;
    let short_input_points = input_points;
    let cycle =
        Stage3bSumchecks(VerifierStage3b::new(&witness.layout, r1.clone(), x.clone()).unwrap());
    let inputs = Stage3bInputClaims {
        variant: RouterCycleVariantInputClaims { fold: folds[0] },
        shift: RouterCycleShiftInputClaims { fold: folds[1] },
        memory: RouterCycleMemoryInputClaims { fold: folds[2] },
        compare: RouterCycleCompareInputClaims { fold: folds[3] },
        branch: RouterCycleBranchInputClaims { fold: folds[4] },
    };
    let input_points = cycle.input_points().unwrap();
    let challenges = cycle.draw_challenges(&mut prover).unwrap();
    let proof = cycle
        .prove(
            &Stage3bKernels::default(),
            &mut ProofSession::default(),
            &mut SequentialRounds,
            witness,
            &inputs,
            &input_points,
            &challenges,
            ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut prover,
        )
        .unwrap();
    let twin = cycle.draw_challenges(&mut verifier).unwrap();
    let points = cycle
        .verify_clear(
            &inputs,
            &input_points,
            &twin,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut verifier,
            0,
        )
        .unwrap();
    cycle.append_output_claims(&mut verifier, &proof.output_claims);
    assert_eq!(prover.state(), verifier.state());
    assert_eq!(points, proof.output_points);
    if let (Some(checked), Some(short_output)) = (checked, &stage3a_output) {
        let wire = BatchProof {
            rounds: proof.recorded.proof.clone(),
            values: stage3b::verify::values(&proof.output_claims),
        };
        let output =
            stage3b::verify::verify(checked, &wire, &mut stage_transcript, short_output).unwrap();
        assert_eq!(output.points, points);
        assert_eq!(stage_transcript.state(), prover.state());
    }
    let r3 = &points.branch.branch;
    let expected_word: Vec<_> = x[..6].iter().chain(r3).copied().collect();
    for actual in [
        &points.variant.rs1_value,
        &points.variant.rs2_value,
        &points.variant.rd_pre_value,
        &points.variant.imm,
        &points.variant.fall_through_pc,
        &points.variant.pc_plus_imm,
        &points.variant.pc,
        &points.variant.next_pc,
        &points.shift.rs1_value,
        &points.memory.rs2_value,
        &points.memory.ram_read_value,
        &points.compare.rs1_value,
        &points.compare.rs2_value,
        &points.compare.imm,
        &points.branch.fall_through_pc,
        &points.branch.pc_plus_imm,
    ] {
        assert_eq!(actual, &expected_word);
    }
    for (actual, prefix) in [
        (&points.variant.variant_bits, &x[..10]),
        (&points.variant.variant, &x[11..17]),
        (&points.shift.shift_kind, &x[12..15]),
        (&points.memory.access_kind, &x[13..17]),
        (&points.compare.key_kind, &x[14..17]),
        (&points.shift.pos_ra_0, &x[6..9]),
        (&points.memory.pos_ra_0, &x[6..9]),
        (&points.compare.pos_ra_0, &x[6..9]),
        (&points.shift.pos_ra_1, &x[9..12]),
        (&points.compare.pos_ra_1, &x[9..12]),
        (&points.branch.branch, &x[..0]),
        (&points.branch.should_branch, &x[..0]),
    ] {
        let expected: Vec<_> = prefix.iter().chain(r3).copied().collect();
        assert_eq!(actual, &expected);
    }
    let eq_cycle = evaluate(
        (0..witness.bits.len())
            .map(|j| eq_index(&r1, j).unwrap())
            .collect(),
        r3,
    );
    macro_rules! check_relation {
        ($member:ident,$router:expr,$fold:expr) => {{
            assert_eq!(
                cycle
                    .$member
                    .input_claim(&inputs.$member, &challenges.$member)
                    .unwrap(),
                $fold
            );
            let src = evaluate(
                (0..witness.bits.len())
                    .map(|j| source_at(witness, $router, &x, j))
                    .collect(),
                r3,
            );
            let expected = eq_cycle * src * terminal_selector(witness, $router, &x, r3);
            assert_eq!(
                cycle
                    .$member
                    .expected_output(
                        &input_points.$member,
                        &proof.output_claims.$member,
                        &points.$member,
                        &challenges.$member
                    )
                    .unwrap(),
                expected
            );
        }};
    }
    check_relation!(variant, Router::Variant, folds[0]);
    check_relation!(shift, Router::Shift, folds[1]);
    check_relation!(memory, Router::Memory, folds[2]);
    check_relation!(compare, Router::Compare, folds[3]);
    check_relation!(branch, Router::Branch, folds[4]);

    let v = &proof.output_claims.variant;
    let s = &proof.output_claims.shift;
    let m = &proof.output_claims.memory;
    let c = &proof.output_claims.compare;
    let b = &proof.output_claims.branch;
    let actual = [
        v.rs1_value,
        v.rs2_value,
        v.rd_pre_value,
        v.imm,
        v.fall_through_pc,
        v.pc_plus_imm,
        v.pc,
        v.next_pc,
        v.variant_bits,
        v.variant,
        s.shift_kind,
        s.pos_ra_0,
        s.pos_ra_1,
        m.ram_read_value,
        m.access_kind,
        c.key_kind,
        b.branch,
        b.should_branch,
    ];
    for (claim, &value) in actual.iter().enumerate() {
        let table = (0..witness.bits.len())
            .map(|j| {
                let row = fetched(witness, j);
                let words = &witness.words[j];
                let word = match claim {
                    0 => Some(words.rs1_value),
                    1 => Some(words.rs2_value),
                    2 => Some(words.rd_pre_value),
                    3 => Some(row.imm),
                    4 => Some(row.fall_through_pc),
                    5 => Some(row.pc_plus_imm),
                    6 => Some(row.pc),
                    7 => Some(words.next_pc),
                    13 => Some(words.ram_read_value),
                    _ => None,
                };
                if let Some(word) = word {
                    return word_at(word, &x[..6]);
                }
                match claim {
                    8 => {
                        let g = usize::from(witness.layout.ram_ra()[0].start());
                        let inc = eq_index(&x[6..10], 8).unwrap()
                            * word_at(witness.layout.inc(&witness.bits[j]), &x[..6]);
                        inc + (g..witness.layout.used_columns())
                            .filter(|&y| bit(&witness.bits[j], y))
                            .map(|y| eq_index(&x[..10], 576 + y - g).unwrap())
                            .sum::<F128>()
                    }
                    9 => kind_at(witness, j, Router::Variant, &x),
                    10 => kind_at(witness, j, Router::Shift, &x),
                    11 => position_at(witness, j, 0, &x),
                    12 => position_at(witness, j, 1, &x),
                    14 => kind_at(witness, j, Router::Memory, &x),
                    15 => kind_at(witness, j, Router::Compare, &x),
                    16 => scalar(row.variant.unwrap().branch().is_some()),
                    17 => scalar(bit(&witness.bits[j], witness.layout.should_branch())),
                    _ => unreachable!(),
                }
            })
            .collect();
        assert_eq!(value, evaluate(table, r3), "wire claim {claim}");
    }
    if (0..witness.bits.len()).any(|j| fetched(witness, j).variant.unwrap().shift().is_some()) {
        for changed in 0..18 {
            let mut transcript = BinaryTranscript::new(b"rv64i-router-batches");
            let short_challenges = short.draw_challenges(&mut transcript).unwrap();
            let _ = short
                .verify_clear(
                    &short_inputs,
                    &short_input_points,
                    &short_challenges,
                    &short_proof.output_claims,
                    &short_proof.recorded.proof,
                    &mut transcript,
                    0,
                )
                .unwrap();
            short.append_output_claims(&mut transcript, &short_proof.output_claims);
            let challenges = cycle.draw_challenges(&mut transcript).unwrap();
            let mut wire = cycle_values(&proof.output_claims);
            let cells = [
                &mut wire.rs1_value,
                &mut wire.rs2_value,
                &mut wire.rd_pre_value,
                &mut wire.imm,
                &mut wire.fall_through_pc,
                &mut wire.pc_plus_imm,
                &mut wire.pc,
                &mut wire.next_pc,
                &mut wire.variant_bits,
                &mut wire.variant,
                &mut wire.shift_kind,
                &mut wire.pos_ra_0,
                &mut wire.pos_ra_1,
                &mut wire.ram_read_value,
                &mut wire.access_kind,
                &mut wire.key_kind,
                &mut wire.branch,
                &mut wire.should_branch,
            ];
            *cells.into_iter().nth(changed).unwrap() += F128::one();
            let changed_claims = expand_cycle(&wire);
            assert!(
                cycle
                    .verify_clear(
                        &inputs,
                        &input_points,
                        &challenges,
                        &changed_claims,
                        &proof.recorded.proof,
                        &mut transcript,
                        0
                    )
                    .is_err(),
                "changed cycle cell {changed}"
            );
        }
    }
}

#[test]
fn synthetic_router_batches_bind_all_eighteen_cycle_values() {
    let (statement, _, source) = support::counting_loop();
    let layout = Layout::new(6, 6, source.layout.lowest_address()).unwrap();
    let instructions = [
        support::asm::sll(1, 1, 1),
        support::asm::srl(1, 1, 1),
        support::asm::sra(1, 1, 1),
        support::asm::sllw(1, 1, 1),
        support::asm::srlw(1, 1, 1),
        support::asm::sraw(1, 1, 1),
        support::asm::slt(1, 1, 1),
        support::asm::sltu(1, 1, 1),
        support::asm::slti(1, 1, 1),
        support::asm::sltiu(1, 1, 1),
        support::asm::lb(1, 1, 0),
        support::asm::lh(1, 1, 0),
        support::asm::lw(1, 1, 0),
        support::asm::ld(1, 1, 0),
        support::asm::lbu(1, 1, 0),
        support::asm::lhu(1, 1, 0),
        support::asm::lwu(1, 1, 0),
        support::asm::sb(1, 1, 0),
        support::asm::sh(1, 1, 0),
        support::asm::sw(1, 1, 0),
        support::asm::sd(1, 1, 0),
        support::asm::beq(1, 1, 0),
        support::asm::bne(1, 1, 0),
        support::asm::blt(1, 1, 0),
        support::asm::bge(1, 1, 0),
        support::asm::bltu(1, 1, 0),
        support::asm::bgeu(1, 1, 0),
        support::asm::add(1, 1, 1),
        support::asm::and(1, 1, 1),
        support::asm::jal(1, 0),
        support::asm::jal(0, 0),
    ];
    let program: Vec<_> = instructions
        .into_iter()
        .enumerate()
        .map(|(i, instruction)| (RAM_START_ADDRESS + 4 * i as u64, instruction))
        .collect();
    let bytecode = Arc::new(support::harness::bytecode(&program, &layout));
    let witness = Rv64iWitness::synthetic(
        0x726f_7574,
        layout,
        bytecode,
        &statement.device.memory_layout,
        source.initial_ram.clone(),
        5,
    )
    .unwrap();
    for router in [
        Router::Shift,
        Router::Memory,
        Router::Compare,
        Router::Branch,
    ] {
        assert!((0..witness.bits.len()).any(|j| match router {
            Router::Shift => fetched(&witness, j).variant.unwrap().shift().is_some(),
            Router::Memory => fetched(&witness, j)
                .variant
                .unwrap()
                .access()
                .and_then(|a| a.kind)
                .is_some(),
            Router::Compare => fetched(&witness, j).variant.unwrap().key_kind().is_some(),
            Router::Branch => fetched(&witness, j).variant.unwrap().branch().is_some(),
            Router::Variant => false,
        }));
    }
    prove_router_batches(&witness, None);
}

#[test]
fn router_batches_accept_interpreter_and_replay_counting_loop() {
    let (statement, preprocessing, source) = support::counting_loop();
    let witness = Rv64iWitness::from_facts(
        source.layout.clone(),
        Arc::clone(&source.bytecode),
        &support::counting_loop_facts(),
        source.initial_ram.clone(),
    )
    .unwrap();
    let checked = CheckedInputs::of_statement(
        &preprocessing,
        &statement,
        witness.layout.log_K_ram() as u8,
        witness.final_pc,
    )
    .unwrap();
    prove_router_batches(&witness, Some(&checked));
}

#[test]
fn matrix_bits_support_is_exactly_the_direct_interval() {
    for (b, a) in [(4, 5), (8, 10), (20, 20), (20, 27)] {
        let layout = Layout::new(b, a, 0).unwrap();
        let matrices = RowSystem::new(&layout).to_matrices();
        let support: BTreeSet<_> = matrices
            .a
            .iter()
            .chain(&matrices.b)
            .chain(&matrices.c)
            .flatten()
            .filter(|(column, value)| *column >= 768 && !value.is_zero())
            .map(|(column, _)| column - 768)
            .collect();
        assert_eq!(
            support,
            (64..=layout.keys_differ()).collect(),
            "layout {b},{a}"
        );
    }
}
