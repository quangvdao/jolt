//! Independent row and column identities for the binary-field RV64I protocol.
#![expect(
    clippy::unwrap_used,
    reason = "invalid test fixtures fail the enclosing test"
)]

mod support;

use jolt_claims::NoChallenges;
use jolt_crypto::NoCommitment;
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{ProofSession, ProverInputs};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_prover::driver::StageProver;
use jolt_r1cs::ConstraintMatrices;
use jolt_rv64i_arith::{Layout, RowSystem, WitnessRow};
use jolt_rv64i_prover::commitment::{transparent::TransparentBits, BitsCommitmentProver};
use jolt_rv64i_prover::plane::Rv64iWitness;
use jolt_rv64i_prover::stages::stage1::{Stage1Kernels, Stage1Sumchecks};
use jolt_rv64i_prover::stages::stage2::{Stage2Kernels, Stage2Sumchecks};
use jolt_rv64i_verifier::claims::spartan_inner::{
    SpartanInnerChallenges, SpartanInnerInputClaims, SpartanInnerOutputClaims,
};
use jolt_rv64i_verifier::claims::spartan_outer::{
    SpartanOuterF128OutputClaims, SpartanOuterF2OutputClaims, SpartanOuterInputClaims,
};
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry};
use jolt_rv64i_verifier::ids::{
    DerivedId, OpeningId, OuterDerived, RelationId, RowBlock, VirtualPolynomial,
};
use jolt_rv64i_verifier::points::{eq_index, to_high_to_low};
use jolt_rv64i_verifier::proof::{BatchProof, InnerValues, OuterValues};
use jolt_rv64i_verifier::public::matrices::RowMatrices;
use jolt_rv64i_verifier::stages::stage1::{
    Stage1InputClaims, Stage1OutputClaims, Stage1Sumchecks as VerifierStage1Sumchecks,
};
use jolt_rv64i_verifier::stages::stage2::Stage2Sumchecks as VerifierStage2Sumchecks;
use jolt_rv64i_verifier::stages::{stage1, stage2};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_rv64i_verifier::transcript::{preamble, Rv64iTranscript};
use jolt_sumcheck::{
    prove_batch, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SequentialRounds,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn random_point(rng: &mut StdRng, variables: usize) -> Vec<F128> {
    (0..variables).map(|_| F128::from_raw(rng.gen())).collect()
}

fn evaluate(table: &[F128], point: &[F128]) -> F128 {
    Polynomial::new(table.to_vec()).evaluate(&to_high_to_low(point))
}

fn witness_rows(witness: &Rv64iWitness) -> Vec<WitnessRow> {
    witness
        .bits
        .iter()
        .enumerate()
        .map(|(j, bits)| {
            let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
            let base = witness.words[j]
                .base_words(row.variant.unwrap().is_store(), witness.layout.inc(bits));
            WitnessRow::compute(&witness.layout, row, &base, bits)
        })
        .collect()
}

fn column(row: &WitnessRow, col: usize) -> F128 {
    F128::from_u64((row.0[col / 64] >> (col % 64)) & 1)
}

fn row_tables(
    matrices: &ConstraintMatrices<F128>,
    rows: &[WitnessRow],
    block: usize,
    variables: usize,
) -> [Vec<F128>; 3] {
    let start = if block == 0 { 0 } else { RowSystem::F2_ROWS };
    let end = if block == 0 {
        RowSystem::F2_ROWS
    } else {
        matrices.num_constraints
    };
    [&matrices.a, &matrices.b, &matrices.c].map(|matrix| {
        rows.iter()
            .flat_map(|row| {
                (0..1 << variables).map(move |i| {
                    if start + i >= end {
                        F128::zero()
                    } else {
                        matrix[start + i]
                            .iter()
                            .map(|&(col, coefficient)| coefficient * column(row, col))
                            .sum()
                    }
                })
            })
            .collect()
    })
}

#[test]
fn matrices_pin_direct_interval_and_reference_nonzero_count() {
    let mut rng = StdRng::seed_from_u64(0x6401_1255);
    for (b, a) in [(4, 5), (8, 10), (20, 20), (20, 27)] {
        let layout = Layout::new(b, a, 0).unwrap();
        let public = RowMatrices::new(&layout);
        let matrices = RowSystem::new(&layout).to_matrices();
        let direct: BTreeSet<_> = [&matrices.a, &matrices.b, &matrices.c]
            .into_iter()
            .flatten()
            .flatten()
            .filter(|(col, coefficient)| *col >= 768 && *coefficient != F128::zero())
            .map(|(col, _)| col - 768)
            .collect();
        assert_eq!(direct, (64..=layout.keys_differ()).collect());
        if (b, a) == (20, 20) {
            let nonzero = [&matrices.a, &matrices.b, &matrices.c]
                .into_iter()
                .flatten()
                .flatten()
                .filter(|(_, coefficient)| *coefficient != F128::zero())
                .count();
            assert_eq!(nonzero, 1255);
            assert_eq!(direct.len(), 165);
        }
        let rho_f2 = random_point(&mut rng, 8);
        let rho_f128 = random_point(&mut rng, public.f128_row_variables());
        let w = random_point(&mut rng, 10);
        let actual = public.evaluate(&rho_f2, &rho_f128, &w).unwrap();
        for (block, rho) in [&rho_f2, &rho_f128].into_iter().enumerate() {
            let start = if block == 0 { 0 } else { RowSystem::F2_ROWS };
            let end = if block == 0 {
                RowSystem::F2_ROWS
            } else {
                matrices.num_constraints
            };
            for (side, matrix) in [&matrices.a, &matrices.b, &matrices.c]
                .into_iter()
                .enumerate()
            {
                let expected: F128 = (start..end)
                    .flat_map(|i| {
                        matrix[i]
                            .iter()
                            .map(move |&(col, coefficient)| (i, col, coefficient))
                    })
                    .map(|(i, col, coefficient)| {
                        eq_index(rho, i - start).unwrap() * eq_index(&w, col).unwrap() * coefficient
                    })
                    .sum();
                assert_eq!(actual.blocks[block][side], expected);
            }
        }
    }
}

#[test]
fn outer_and_inner_expressions_equal_cube_definitions() {
    let (statement, _, source) = support::counting_loop();
    let facts = support::counting_loop_facts();
    let witness = Rv64iWitness::from_facts(
        source.layout.clone(),
        Arc::clone(&source.bytecode),
        &facts[..32],
        source.initial_ram.clone(),
    )
    .unwrap();
    let rows = witness_rows(&witness);
    let matrices = Arc::new(RowMatrices::new(&witness.layout));
    let m_f = matrices.f128_row_variables();
    let mut rng = StdRng::seed_from_u64(0x6401_e8e2);
    let tau_f2 = random_point(&mut rng, 13);
    let tau_f128 = random_point(&mut rng, m_f + 5);
    let outer = VerifierStage1Sumchecks::new(5, m_f, tau_f2.clone(), tau_f128.clone()).unwrap();
    let p = random_point(&mut rng, 13);
    let tables_f2 = row_tables(matrices.matrices(), &rows, 0, 8);
    let tables_f128 = row_tables(matrices.matrices(), &rows, 1, m_f);
    let inputs = SpartanOuterInputClaims::default();
    let points = SpartanOuterInputClaims::default();
    let challenges = NoChallenges::default();
    let values_f2 = tables_f2.each_ref().map(|table| evaluate(table, &p));
    let values_f128 = tables_f128
        .each_ref()
        .map(|table| evaluate(table, &p[8 - m_f..]));
    let output_f2 = SpartanOuterF2OutputClaims {
        az: values_f2[0],
        bz: values_f2[1],
        cz: values_f2[2],
    };
    let output_f128 = SpartanOuterF128OutputClaims {
        az: values_f128[0],
        bz: values_f128[1],
        cz: values_f128[2],
    };
    let points_f2 = outer
        .spartan_outer_f2
        .derive_opening_points(&p, &points)
        .unwrap();
    let points_f128 = outer
        .spartan_outer_f128
        .derive_opening_points(&p[8 - m_f..], &points)
        .unwrap();
    for (tau, tables) in [(&tau_f2, &tables_f2), (&tau_f128, &tables_f128)] {
        let sum: F128 = (0..tables[0].len())
            .map(|index| {
                eq_index(tau, index).unwrap()
                    * (tables[0][index] * tables[1][index] + tables[2][index])
            })
            .sum();
        assert_eq!(sum, F128::zero());
    }
    assert_eq!(
        outer
            .spartan_outer_f2
            .input_claim(&inputs, &challenges)
            .unwrap(),
        F128::zero()
    );
    assert_eq!(
        outer
            .spartan_outer_f128
            .input_claim(&inputs, &challenges)
            .unwrap(),
        F128::zero()
    );
    let eq_f2: Vec<_> = (0..1 << 13)
        .map(|i| eq_index(&tau_f2, i).unwrap())
        .collect();
    let eq_f128: Vec<_> = (0..1 << (m_f + 5))
        .map(|i| eq_index(&tau_f128, i).unwrap())
        .collect();
    assert_eq!(
        outer
            .spartan_outer_f2
            .expected_output(&points, &output_f2, &points_f2, &challenges)
            .unwrap(),
        evaluate(&eq_f2, &p) * (values_f2[0] * values_f2[1] + values_f2[2])
    );
    assert_eq!(
        outer
            .spartan_outer_f128
            .expected_output(&points, &output_f128, &points_f128, &challenges)
            .unwrap(),
        evaluate(&eq_f128, &p[8 - m_f..]) * (values_f128[0] * values_f128[1] + values_f128[2])
    );
    let witness = Rv64iWitness::synthetic(
        0x6401_51e5,
        source.layout.clone(),
        Arc::clone(&source.bytecode),
        &statement.device.memory_layout,
        source.initial_ram.clone(),
        5,
    )
    .unwrap();
    let rows = witness_rows(&witness);
    let tables_f2 = row_tables(matrices.matrices(), &rows, 0, 8);
    let tables_f128 = row_tables(matrices.matrices(), &rows, 1, m_f);
    let values_f2 = tables_f2.each_ref().map(|table| evaluate(table, &p));
    let values_f128 = tables_f128
        .each_ref()
        .map(|table| evaluate(table, &p[8 - m_f..]));
    let rho_f2 = &p[..8];
    let rho_f128 = &p[8 - m_f..8];
    let r_1 = &p[8..];
    let inner = VerifierStage2Sumchecks::new(
        Arc::clone(&matrices),
        rho_f2.to_vec(),
        rho_f128.to_vec(),
        r_1.to_vec(),
    )
    .unwrap();
    let coefficients = random_point(&mut rng, 6);
    let c = SpartanInnerChallenges {
        az_f2: coefficients[0],
        bz_f2: coefficients[1],
        cz_f2: coefficients[2],
        az_f128: coefficients[3],
        bz_f128: coefficients[4],
        cz_f128: coefficients[5],
    };
    let input = SpartanInnerInputClaims {
        az_f2: values_f2[0],
        bz_f2: values_f2[1],
        cz_f2: values_f2[2],
        az_f128: values_f128[0],
        bz_f128: values_f128[1],
        cz_f128: values_f128[2],
    };
    let input_points = SpartanInnerInputClaims {
        az_f2: p.clone(),
        bz_f2: p.clone(),
        cz_f2: p.clone(),
        az_f128: p[8 - m_f..].to_vec(),
        bz_f128: p[8 - m_f..].to_vec(),
        cz_f128: p[8 - m_f..].to_vec(),
    };
    let mut matrix_weight = vec![F128::zero(); 1024];
    for (block, rho) in [rho_f2, rho_f128].into_iter().enumerate() {
        let start = if block == 0 { 0 } else { RowSystem::F2_ROWS };
        let end = if block == 0 {
            RowSystem::F2_ROWS
        } else {
            matrices.matrices().num_constraints
        };
        for (side, matrix) in [
            &matrices.matrices().a,
            &matrices.matrices().b,
            &matrices.matrices().c,
        ]
        .into_iter()
        .enumerate()
        {
            for (i, row) in matrix.iter().enumerate().take(end).skip(start) {
                for &(col, coefficient) in row {
                    matrix_weight[col] += coefficients[3 * block + side]
                        * eq_index(rho, i - start).unwrap()
                        * coefficient;
                }
            }
        }
    }
    let mut routed = vec![F128::zero(); 1024];
    let mut direct = vec![F128::zero(); 1024];
    for col in (1..4).chain(16..27).chain(64..768) {
        let table: Vec<_> = rows
            .iter()
            .map(|row| column(row, col) + F128::from_u64(u64::from(col == 16)))
            .collect();
        routed[col] = evaluate(&table, r_1);
    }
    for (col, value) in direct
        .iter_mut()
        .enumerate()
        .take(769 + witness.layout.keys_differ())
        .skip(832)
    {
        let table: Vec<_> = rows.iter().map(|row| column(row, col)).collect();
        *value = evaluate(&table, r_1);
    }
    let mut public = vec![F128::zero(); 1024];
    public[0] = F128::one();
    public[16] = F128::one();
    let direct_sum: F128 = (0..1024)
        .map(|col| matrix_weight[col] * (routed[col] + direct[col] + public[col]))
        .sum();
    assert_eq!(
        inner.spartan_inner.input_claim(&input, &c).unwrap(),
        direct_sum
    );
    let w = random_point(&mut rng, 10);
    let output = SpartanInnerOutputClaims {
        witness_routed: evaluate(&routed, &w),
        direct_columns: evaluate(&direct, &w),
    };
    let output_points = inner
        .spartan_inner
        .derive_opening_points(&w, &input_points)
        .unwrap();
    assert_eq!(
        inner
            .spartan_inner
            .expected_output(&input_points, &output, &output_points, &c)
            .unwrap(),
        evaluate(&matrix_weight, &w)
            * (evaluate(&routed, &w) + evaluate(&direct, &w) + evaluate(&public, &w))
    );
}

#[test]
fn batch_one_places_dense_members_on_a_shared_cycle_suffix() {
    let layout = Layout::new(4, 6, 0).unwrap();
    let matrices = RowMatrices::new(&layout);
    let m_f = matrices.f128_row_variables();
    let mut rng = StdRng::seed_from_u64(0x6401_0001);
    let tau_f2 = random_point(&mut rng, 11);
    let tau_f128 = random_point(&mut rng, m_f + 3);
    let batch = VerifierStage1Sumchecks::new(3, m_f, tau_f2.clone(), tau_f128.clone()).unwrap();
    let inputs = Stage1InputClaims {
        spartan_outer_f2: SpartanOuterInputClaims::default(),
        spartan_outer_f128: SpartanOuterInputClaims::default(),
    };
    let input_points = batch.empty_input_points();
    let mut transcript = Rv64iTranscript::new(b"rv64i-batch-one-placement");
    let challenges = batch.draw_challenges(&mut transcript).unwrap();
    let mut recorder = ClearSumcheckRecorder::<F128, NoCommitment>::new();
    let (prelude, coefficients) = batch
        .begin_batch(&inputs, &challenges, &mut recorder, &mut transcript)
        .unwrap();
    assert_eq!(prelude.max_num_vars, 11);
    assert_eq!(prelude.max_degree, 3);
    assert_eq!(
        (prelude.members[0].offset, prelude.members[0].rounds),
        (0, 11)
    );
    assert_eq!(
        (prelude.members[1].offset, prelude.members[1].rounds),
        (8 - m_f, m_f + 3)
    );
    let tables_f2: [Vec<F128>; 3] = std::array::from_fn(|_| random_point(&mut rng, 1 << 11));
    let tables_f128: [Vec<F128>; 3] =
        std::array::from_fn(|_| random_point(&mut rng, 1 << (m_f + 3)));
    let eq_f2: Vec<_> = (0..1 << 11)
        .map(|i| eq_index(&tau_f2, i).unwrap())
        .collect();
    let eq_f128: Vec<_> = (0..1 << (m_f + 3))
        .map(|i| eq_index(&tau_f128, i).unwrap())
        .collect();
    macro_rules! kernel {
        ($member:ident, $tables:ident, $eq:ident, $relation:ident, $block:ident) => {{
            let openings = [
                VirtualPolynomial::Az,
                VirtualPolynomial::Bz,
                VirtualPolynomial::Cz,
            ]
            .into_iter()
            .zip($tables.each_ref())
            .map(|(p, table)| {
                (
                    OpeningId::virtual_polynomial(p, RelationId::$relation),
                    Polynomial::new(table.clone()),
                )
            })
            .collect();
            let derived = BTreeMap::from([(
                DerivedId::SpartanOuter(RowBlock::$block, OuterDerived::EqTau),
                Polynomial::new($eq.clone()),
            )]);
            let inputs = ProverInputs {
                relation: &batch.$member,
                claims: &inputs.$member,
                points: &input_points.$member,
                challenges: &challenges.$member,
            };
            NaiveSumcheckProver::new(&inputs, openings, derived, BindingOrder::LowToHigh).unwrap()
        }};
    }
    let mut f2 = kernel!(spartan_outer_f2, tables_f2, eq_f2, SpartanOuterF2, F2);
    let mut f128 = kernel!(
        spartan_outer_f128,
        tables_f128,
        eq_f128,
        SpartanOuterF128,
        F128
    );
    let mut members = prelude.members;
    for ((member, tables), eq) in members
        .iter_mut()
        .zip([&tables_f2, &tables_f128])
        .zip([&eq_f2, &eq_f128])
    {
        member.input_claim = (0..eq.len())
            .map(|i| eq[i] * (tables[0][i] * tables[1][i] + tables[2][i]))
            .sum();
    }
    let prelude = BatchPrelude::new(members, 11, 3);
    let mut kernels: Vec<&mut dyn ProveRounds<F128>> = vec![&mut f2, &mut f128];
    let proved = prove_batch(
        &prelude,
        &mut kernels,
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .unwrap();
    let p = &proved.challenges;
    let p_f128 = &p[8 - m_f..];
    let values_f2 = tables_f2.each_ref().map(|table| evaluate(table, p));
    let values_f128 = tables_f128.each_ref().map(|table| evaluate(table, p_f128));
    let outputs = Stage1OutputClaims {
        spartan_outer_f2: SpartanOuterF2OutputClaims {
            az: values_f2[0],
            bz: values_f2[1],
            cz: values_f2[2],
        },
        spartan_outer_f128: SpartanOuterF128OutputClaims {
            az: values_f128[0],
            bz: values_f128[1],
            cz: values_f128[2],
        },
    };
    let output_points = batch.derive_opening_points(p, &input_points).unwrap();
    assert_eq!(
        &output_points.spartan_outer_f2.az[8..],
        &output_points.spartan_outer_f128.az[m_f..]
    );
    let lambda: F128 = p[..8 - m_f].iter().map(|z| F128::one() + *z).product();
    let terminal_f2 = evaluate(&eq_f2, p) * (values_f2[0] * values_f2[1] + values_f2[2]);
    let terminal_f128 =
        evaluate(&eq_f128, p_f128) * (values_f128[0] * values_f128[1] + values_f128[2]);
    let expected = coefficients.spartan_outer_f2 * terminal_f2
        + coefficients.spartan_outer_f128 * lambda * terminal_f128;
    assert_eq!(proved.final_claim, expected);
    assert_eq!(
        batch
            .expected_final_claim(
                &prelude,
                p,
                &coefficients,
                &input_points,
                &outputs,
                &output_points,
                &challenges
            )
            .unwrap(),
        expected
    );
}

#[test]
fn counting_loop_batches_one_and_two_reject_each_wire_value() {
    let (statement, preprocessing, witness) = support::counting_loop();
    let checked = CheckedInputs::of_statement(
        &preprocessing,
        &statement,
        witness.layout.log_K_ram() as u8,
        witness.final_pc,
    )
    .unwrap();
    let geometry = BitsGeometry { log_T: 6 };
    let mut transcript: Rv64iTranscript = preamble(&checked);
    let (commitment, _) =
        TransparentBits::commit(&(), geometry, &witness.bits, &mut transcript).unwrap();
    let start_verifier = || {
        let mut transcript: Rv64iTranscript = preamble(&checked);
        let _state =
            TransparentBits::verify_commit(&(), geometry, &commitment, &mut transcript).unwrap();
        transcript
    };
    let matrices = Arc::new(RowMatrices::new(&witness.layout));
    let m_f = matrices.f128_row_variables();
    let tau_f2 = transcript.challenge_vector(14);
    let tau_f128 = transcript.challenge_vector(m_f + 6);
    let batch1 = Stage1Sumchecks(VerifierStage1Sumchecks::new(6, m_f, tau_f2, tau_f128).unwrap());
    let inputs1 = Stage1InputClaims {
        spartan_outer_f2: SpartanOuterInputClaims::default(),
        spartan_outer_f128: SpartanOuterInputClaims::default(),
    };
    let points1 = batch1.empty_input_points();
    let challenges1 = batch1.draw_challenges(&mut transcript).unwrap();
    let proved1 = batch1
        .prove(
            &Stage1Kernels::default(),
            &mut ProofSession::default(),
            &mut SequentialRounds,
            &witness,
            &inputs1,
            &points1,
            &challenges1,
            ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut transcript,
        )
        .unwrap();
    let a = &proved1.output_claims.spartan_outer_f2;
    let b = &proved1.output_claims.spartan_outer_f128;
    let proof1 = BatchProof {
        rounds: proved1.recorded.proof,
        values: OuterValues {
            az_f2: a.az,
            bz_f2: a.bz,
            cz_f2: a.cz,
            az_f128: b.az,
            bz_f128: b.bz,
            cz_f128: b.cz,
        },
    };
    let mut verifier_transcript = start_verifier();
    let verified1 = stage1::verify::verify(&checked, &proof1, &mut verifier_transcript).unwrap();
    assert_eq!(verifier_transcript.state(), transcript.state());
    let stage2::verify::Inputs {
        batch: batch2,
        claims: inputs2,
        points: points2,
    } = stage2::verify::from_upstream(Arc::clone(&matrices), &verified1).unwrap();
    let batch2 = Stage2Sumchecks(batch2);
    let challenges2 = batch2.draw_challenges(&mut transcript).unwrap();
    let proved2 = batch2
        .prove(
            &Stage2Kernels::default(),
            &mut ProofSession::default(),
            &mut SequentialRounds,
            &witness,
            &inputs2,
            &points2,
            &challenges2,
            ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut transcript,
        )
        .unwrap();
    let values = &proved2.output_claims.spartan_inner;
    let proof2 = BatchProof {
        rounds: proved2.recorded.proof,
        values: InnerValues {
            witness_routed: values.witness_routed,
            direct_columns: values.direct_columns,
        },
    };
    let verified2 =
        stage2::verify::verify(&checked, &proof2, &mut verifier_transcript, &verified1).unwrap();
    assert_eq!(verifier_transcript.state(), transcript.state());
    let rows = witness_rows(&witness);
    let r_1 = verified1.r_1().unwrap();
    let w = verified2.w().unwrap();
    let expected_routed: F128 = (1..4)
        .chain(16..27)
        .chain(64..768)
        .map(|col| {
            let table: Vec<_> = rows
                .iter()
                .map(|row| column(row, col) + F128::from_u64(u64::from(col == 16)))
                .collect();
            eq_index(w, col).unwrap() * evaluate(&table, r_1)
        })
        .sum();
    let expected_direct: F128 = (64..=witness.layout.keys_differ())
        .map(|col| {
            let table: Vec<_> = rows.iter().map(|row| column(row, 768 + col)).collect();
            eq_index(w, 768 + col).unwrap() * evaluate(&table, r_1)
        })
        .sum();
    assert_eq!(proof2.values.witness_routed, expected_routed);
    assert_eq!(proof2.values.direct_columns, expected_direct);
    for changed in 0..8 {
        let mut proof1 = proof1.clone();
        let mut proof2 = proof2.clone();
        let value = match changed {
            0 => &mut proof1.values.az_f2,
            1 => &mut proof1.values.bz_f2,
            2 => &mut proof1.values.cz_f2,
            3 => &mut proof1.values.az_f128,
            4 => &mut proof1.values.bz_f128,
            5 => &mut proof1.values.cz_f128,
            6 => &mut proof2.values.witness_routed,
            7 => &mut proof2.values.direct_columns,
            _ => unreachable!(),
        };
        *value += F128::one();
        let mut transcript = start_verifier();
        match stage1::verify::verify(&checked, &proof1, &mut transcript) {
            Ok(upstream) => assert!(
                stage2::verify::verify(&checked, &proof2, &mut transcript, &upstream).is_err(),
                "changed wire value {changed}"
            ),
            Err(_) => assert!(changed < 6, "unchanged outer proof must verify"),
        }
    }
}
