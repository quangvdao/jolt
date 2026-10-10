//! Dense test oracles for register reads, RAM reads and final public I/O.

use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::views::{self, RegisterSelector};
use jolt_field::{Ring, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_verifier::ids::{
    DerivedId, OpeningId, OutputCheckDerived, ReadCheckingDerived, RelationId, VirtualPolynomial,
};
use jolt_rv64i_verifier::points::{eq_index, WordLift};
use jolt_rv64i_verifier::public::io::word_at;
use jolt_rv64i_verifier::stages::stage4::{
    ram_output_check::RamOutputCheck, ram_read_checking::RamReadChecking,
    registers_read_checking::RegistersReadChecking,
};
use std::collections::BTreeMap;
use std::fmt::Display;

fn geometry(error: impl Display) -> KernelError<F128> {
    KernelError::InvalidGeometry {
        reason: error.to_string(),
    }
}

#[derive(Default)]
pub struct RegistersReadCheckingPrepare;
impl PrepareKernel<F128, RegistersReadChecking<F128>, Rv64iPlane> for RegistersReadCheckingPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RegistersReadChecking<F128>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F128, Relation = RegistersReadChecking<F128>>>,
        KernelError<F128>,
    > {
        let openings = [
            (
                VirtualPolynomial::Rs1Ra,
                views::register_selector(witness, RegisterSelector::Rs1).map_err(geometry)?,
            ),
            (
                VirtualPolynomial::Rs2Ra,
                views::register_selector(witness, RegisterSelector::Rs2).map_err(geometry)?,
            ),
            (
                VirtualPolynomial::RdWa,
                views::register_selector(witness, RegisterSelector::Rd).map_err(geometry)?,
            ),
            (
                VirtualPolynomial::RegistersVal,
                views::registers_val(witness, inputs.relation.r_bit()).map_err(geometry)?,
            ),
        ]
        .into_iter()
        .map(|(polynomial, table)| {
            (
                OpeningId::virtual_polynomial(polynomial, RelationId::RegistersReadChecking),
                table,
            )
        })
        .collect();
        let weights = (0..witness.bits.len())
            .map(|j| eq_index(inputs.relation.r_3(), j).map_err(geometry))
            .collect::<Result<Vec<_>, _>>()?;
        let table = weights
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 32))
            .collect();
        let derived = [(
            DerivedId::RegistersReadChecking(ReadCheckingDerived::EqCycle),
            Polynomial::new(table),
        )]
        .into_iter()
        .collect();
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}

#[derive(Default)]
pub struct RamReadCheckingPrepare;
impl PrepareKernel<F128, RamReadChecking<F128>, Rv64iPlane> for RamReadCheckingPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RamReadChecking<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RamReadChecking<F128>>>, KernelError<F128>>
    {
        let count = 1_usize
            .checked_shl(u32::try_from(witness.layout.log_K_ram()).map_err(geometry)?)
            .ok_or_else(|| geometry("RAM domain cannot be represented"))?;
        let openings = [
            (
                VirtualPolynomial::RamRa,
                views::ram_ra(witness).map_err(geometry)?,
            ),
            (
                VirtualPolynomial::RamVal,
                views::ram_val(witness, inputs.relation.r_bit()).map_err(geometry)?,
            ),
        ]
        .into_iter()
        .map(|(polynomial, table)| {
            (
                OpeningId::virtual_polynomial(polynomial, RelationId::RamReadChecking),
                table,
            )
        })
        .collect();
        let weights = (0..witness.bits.len())
            .map(|j| eq_index(inputs.relation.r_3(), j).map_err(geometry))
            .collect::<Result<Vec<_>, _>>()?;
        let table = weights
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, count))
            .collect();
        let derived = [(
            DerivedId::RamReadChecking(ReadCheckingDerived::EqCycle),
            Polynomial::new(table),
        )]
        .into_iter()
        .collect();
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}

#[derive(Default)]
pub struct RamOutputCheckPrepare;
impl PrepareKernel<F128, RamOutputCheck<F128>, Rv64iPlane> for RamOutputCheckPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RamOutputCheck<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RamOutputCheck<F128>>>, KernelError<F128>>
    {
        let lift = WordLift::new(inputs.relation.r_bit()).map_err(geometry)?;
        let final_ram = views::ram_val_final_with_lift(witness, &lift).map_err(geometry)?;
        let count = final_ram.len();
        let openings = [(
            OpeningId::virtual_polynomial(
                VirtualPolynomial::RamValFinal,
                RelationId::RamOutputCheck,
            ),
            final_ram,
        )]
        .into_iter()
        .collect();
        let io = inputs.relation.io();
        let weights = (0..count)
            .map(|k| eq_index(inputs.relation.tau(), k).map_err(geometry))
            .collect::<Result<Vec<_>, _>>()?;
        let mask = (0..count)
            .map(|k| {
                F128::from_u64(u64::from(
                    io.io_mask_start <= k as u128 && (k as u128) < io.io_mask_end,
                ))
            })
            .collect();
        let values = (0..count).map(|k| lift.evaluate(word_at(io, k))).collect();
        let derived: BTreeMap<_, _> = [
            (
                DerivedId::RamOutputCheck(OutputCheckDerived::EqTau),
                Polynomial::new(weights),
            ),
            (
                DerivedId::RamOutputCheck(OutputCheckDerived::IoMask),
                Polynomial::new(mask),
            ),
            (
                DerivedId::RamOutputCheck(OutputCheckDerived::ValIo),
                Polynomial::new(values),
            ),
        ]
        .into_iter()
        .collect();
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}
