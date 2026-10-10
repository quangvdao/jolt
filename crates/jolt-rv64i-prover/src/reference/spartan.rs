//! Dense test oracles for the row zero-checks and witness-column reduction.

use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use jolt_claims::{InputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_arith::{RowSystem, WitnessRow, WITNESS_COLUMNS};
use jolt_rv64i_verifier::claims::spartan_inner::{SpartanInnerInputClaims, SpartanInnerSymbolic};
use jolt_rv64i_verifier::ids::{
    CommittedPolynomial, DerivedId, InnerDerived, OpeningId, OuterDerived, RelationId, RowBlock,
    VirtualPolynomial,
};
use jolt_rv64i_verifier::points::{equality_table, PointsError};
use jolt_rv64i_verifier::public::matrices::RowMatrices;
use jolt_rv64i_verifier::stages::stage1::{SpartanOuterF128, SpartanOuterF2};
use jolt_rv64i_verifier::stages::stage2::SpartanInner;
use std::collections::BTreeMap;
use std::fmt::Display;

fn row(witness: &Rv64iWitness, cycle: usize) -> Result<WitnessRow, Rv64iProverError> {
    let bits = witness
        .bits
        .get(cycle)
        .ok_or(Rv64iProverError::CycleIndex {
            cycle,
            rows: witness.bits.len(),
        })?;
    let index = witness.layout.bytecode_index(bits);
    let fetched = usize::try_from(index)
        .ok()
        .and_then(|i| witness.bytecode.rows().get(i))
        .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
    let variant = fetched
        .variant
        .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
    let words = witness
        .words
        .get(cycle)
        .ok_or(Rv64iProverError::CycleIndex {
            cycle,
            rows: witness.words.len(),
        })?;
    let base = words.base_words(variant.is_store(), witness.layout.inc(bits));
    Ok(WitnessRow::compute(&witness.layout, fetched, &base, bits))
}

/// A witness column is extended in the cycle variables without copying any
/// stored witness table; `WitnessRow::compute` owns the derived columns.
pub fn witness_column(
    witness: &Rv64iWitness,
    column: usize,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                let z = row(witness, j)?;
                let value = z.bit(column).ok_or(PointsError::MissingColumn { column })?;
                Ok(F128::from_u64(u64::from(value)))
            })
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}

fn geometry(error: impl Display) -> KernelError<F128> {
    KernelError::InvalidGeometry {
        reason: error.to_string(),
    }
}

fn outer_tables(
    witness: &Rv64iWitness,
    block: RowBlock,
    row_variables: usize,
) -> Result<[Polynomial<F128>; 3], KernelError<F128>> {
    let matrices = RowMatrices::new(&witness.layout);
    let matrices = matrices.matrices();
    let (start, count) = match block {
        RowBlock::F2 => (0, RowSystem::F2_ROWS),
        RowBlock::F128 => (
            RowSystem::F2_ROWS,
            matrices.num_constraints - RowSystem::F2_ROWS,
        ),
    };
    let width = 1_usize
        .checked_shl(u32::try_from(row_variables).map_err(geometry)?)
        .ok_or_else(|| geometry("outer row dimension is not representable"))?;
    if count > width {
        return Err(geometry("outer row block exceeds its cube"));
    }
    let size = width
        .checked_mul(witness.bits.len())
        .ok_or_else(|| geometry("outer cube is not representable"))?;
    let mut tables: [Vec<F128>; 3] = std::array::from_fn(|_| vec![F128::zero(); size]);
    for j in 0..witness.bits.len() {
        let z = row(witness, j).map_err(geometry)?;
        for (table, matrix) in tables
            .iter_mut()
            .zip([&matrices.a, &matrices.b, &matrices.c])
        {
            for (i, sparse) in matrix.iter().skip(start).take(count).enumerate() {
                let value = sparse.iter().try_fold(
                    F128::zero(),
                    |sum, &(c, coefficient)| -> Result<F128, KernelError<F128>> {
                        let bit = z
                            .bit(c)
                            .ok_or_else(|| geometry("matrix column is outside the witness"))?;
                        Ok(sum + if bit { coefficient } else { F128::zero() })
                    },
                )?;
                let entry = table
                    .get_mut(j * width + i)
                    .ok_or_else(|| geometry("outer cube index exceeds its table"))?;
                *entry = value;
            }
        }
    }
    Ok(tables.map(Polynomial::new))
}

macro_rules! prepare_outer {
    ($prepare:ident, $relation:ident, $id:ident) => {
        #[derive(Default)]
        pub struct $prepare;
        impl PrepareKernel<F128, $relation<F128>, Rv64iPlane> for $prepare {
            fn prepare(&self, _session: &mut ProofSession, witness: &Rv64iWitness, inputs: ProverInputs<'_, F128, $relation<F128>>) -> Result<Box<dyn SumcheckKernel<F128, Relation = $relation<F128>>>, KernelError<F128>> {
                let tables = outer_tables(witness, inputs.relation.block(), inputs.relation.row_variables())?;
                let openings = [VirtualPolynomial::Az, VirtualPolynomial::Bz, VirtualPolynomial::Cz].into_iter().zip(tables).map(|(p, table)| (OpeningId::virtual_polynomial(p, RelationId::$id), table)).collect();
                let eq = equality_table(inputs.relation.tau()).map_err(|error| geometry(&error.to_string()))?;
                let derived = BTreeMap::from([(DerivedId::SpartanOuter(inputs.relation.block(), OuterDerived::EqTau), Polynomial::new(eq))]);
                Ok(Box::new(NaiveSumcheckProver::new(&inputs, openings, derived, BindingOrder::LowToHigh)?))
            }
        }
    };
}
prepare_outer!(SpartanOuterF2Prepare, SpartanOuterF2, SpartanOuterF2);
prepare_outer!(SpartanOuterF128Prepare, SpartanOuterF128, SpartanOuterF128);

#[derive(Default)]
pub struct SpartanInnerPrepare;
impl PrepareKernel<F128, SpartanInner<F128>, Rv64iPlane> for SpartanInnerPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, SpartanInner<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = SpartanInner<F128>>>, KernelError<F128>>
    {
        let weights = equality_table(inputs.relation.r_1()).map_err(geometry)?;
        if weights.len() != witness.bits.len() {
            return Err(geometry("cycle point does not match the witness length"));
        }
        let mut routed = vec![F128::zero(); WITNESS_COLUMNS];
        let mut direct = vec![F128::zero(); WITNESS_COLUMNS];
        for (cycle, (bits, weight)) in witness.bits.iter().zip(weights).enumerate() {
            let z = row(witness, cycle).map_err(geometry)?;
            for column in (1..4).chain(16..27).chain(64..768) {
                if z.bit(column)
                    .ok_or_else(|| geometry("routed column exceeds the witness domain"))?
                {
                    *routed
                        .get_mut(column)
                        .ok_or_else(|| geometry("routed column exceeds the witness domain"))? +=
                        weight;
                }
            }
            for column in 64..=witness.layout.keys_differ() {
                if bits
                    .get(column / 64)
                    .is_some_and(|word| (word >> (column % 64)) & 1 != 0)
                {
                    *direct
                        .get_mut(768 + column)
                        .ok_or_else(|| geometry("direct column exceeds the witness domain"))? +=
                        weight;
                }
            }
        }
        *routed
            .get_mut(16)
            .ok_or_else(|| geometry("routed column exceeds the witness domain"))? += F128::one();
        let matrices = RowMatrices::new(&witness.layout);
        let rho_f2 = inputs
            .points
            .az_f2
            .get(..8)
            .ok_or_else(|| geometry("binary row point is missing its row coordinates"))?;
        let rho_f128 = inputs
            .points
            .az_f128
            .get(..matrices.f128_row_variables())
            .ok_or_else(|| geometry("extension row point is missing its row coordinates"))?;
        let expression = SpartanInnerSymbolic.input_expression::<F128>();
        let matrix = matrices
            .column_evaluations(rho_f2, rho_f128)
            .map_err(geometry)?
            .into_iter()
            .map(|[[az_f2, bz_f2, cz_f2], [az_f128, bz_f128, cz_f128]]| {
                let columns = SpartanInnerInputClaims {
                    az_f2,
                    bz_f2,
                    cz_f2,
                    az_f128,
                    bz_f128,
                    cz_f128,
                };
                expression.try_evaluate(
                    |id| {
                        columns
                            .resolve_input(id)
                            .ok_or_else(|| geometry("matrix fold references an absent row block"))
                    },
                    |id| {
                        inputs
                            .challenges
                            .resolve_challenge(id)
                            .ok_or_else(|| geometry("matrix fold references an absent challenge"))
                    },
                    |_| {
                        Err(geometry(
                            "matrix fold references an unsupported derived term",
                        ))
                    },
                )
            })
            .collect::<Result<Vec<_>, KernelError<F128>>>()?;
        let public = (0..WITNESS_COLUMNS)
            .map(|c| F128::from_u64(u64::from(c == 0 || c == 16)))
            .collect();
        let openings = BTreeMap::from([
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::WitnessRouted,
                    RelationId::SpartanInner,
                ),
                Polynomial::new(routed),
            ),
            (
                OpeningId::committed(CommittedPolynomial::DirectColumns, RelationId::SpartanInner),
                Polynomial::new(direct),
            ),
        ]);
        let derived = BTreeMap::from([
            (
                DerivedId::SpartanInner(InnerDerived::MatrixWeight),
                Polynomial::new(matrix),
            ),
            (
                DerivedId::SpartanInner(InnerDerived::PublicColumns),
                Polynomial::new(public),
            ),
        ]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}
