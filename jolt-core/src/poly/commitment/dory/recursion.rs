//! Jolt implementation of Dory's recursion backend

use ark_bn254::{Fq, Fq12, Fr, G1Affine, G2Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::verify_recursive;
use dory::{
    backends::arkworks::{ArkG1, ArkG2, ArkGT, BN254},
    primitives::arithmetic::{Group, PairingCurve},
    recursion::{
        ast::AstGraph, precompute_challenges, TraceContext, WitnessBackend, WitnessCollection,
        WitnessGenerator, WitnessResult,
    },
};
use std::{collections::HashMap, marker::PhantomData, rc::Rc};

#[cfg(feature = "experimental-pairing-recursion")]
use super::witness::multi_miller_loop::MultiMillerLoopSteps;
use super::{
    commitment_scheme::DoryCommitmentScheme,
    jolt_dory_routines::{JoltG1Routines, JoltG2Routines},
    witness::{
        g1_add::G1AdditionSteps, g1_scalar_mul::ScalarMultiplicationSteps, g2_add::G2AdditionSteps,
        g2_scalar_mul::G2ScalarMultiplicationSteps, gt_exp::Base4ExponentiationSteps,
        gt_mul::MultiplicationSteps,
    },
    wrappers::{
        ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkworksVerifierSetup, JoltToDoryTranscript,
    },
};
use crate::poly::commitment::commitment_scheme::RecursionExt;
use crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::proof_serialization::PairingBoundary;
use crate::zkvm::recursion::witness::{GTCombineWitness, GTExpOpWitness, GTMulOpWitness};
use crate::zkvm::recursion::CombineDag;

/// Jolt witness backend implementation for dory recursion
#[derive(Debug, Clone)]
pub struct JoltWitness;

// `verify_recursive` is fundamentally sequential (transcript + dependency order), but the *witness
// payload construction* for Jolt's recursion SNARK is often more expensive than the underlying
// group operations. To speed up the prover, we first record only minimal per-op inputs/outputs
// during `verify_recursive`, then expand those records into full recursion witnesses in parallel.

#[derive(Debug, Clone)]
struct DeferredJoltWitness;

#[derive(Clone, Debug)]
struct DeferredGtExpWitness {
    base: Fq12,
    exponent: Fr,
    result: Fq12,
    ark_result: ArkGT,
}

impl WitnessResult<ArkGT> for DeferredGtExpWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

#[derive(Clone, Debug)]
struct DeferredGtMulWitness {
    lhs: Fq12,
    rhs: Fq12,
    result: Fq12,
    ark_result: ArkGT,
}

impl WitnessResult<ArkGT> for DeferredGtMulWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

#[derive(Clone, Debug)]
struct DeferredG1ScalarMulWitness {
    point_base: G1Affine,
    scalar: Fr,
    result: G1Affine,
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for DeferredG1ScalarMulWitness {
    fn result(&self) -> Option<&ArkG1> {
        Some(&self.ark_result)
    }
}

#[derive(Clone, Debug)]
struct DeferredG2ScalarMulWitness {
    point_base: G2Affine,
    scalar: Fr,
    result: G2Affine,
    ark_result: ArkG2,
}

impl WitnessResult<ArkG2> for DeferredG2ScalarMulWitness {
    fn result(&self) -> Option<&ArkG2> {
        Some(&self.ark_result)
    }
}

#[derive(Clone, Debug)]
struct DeferredG1AddWitness {
    a: ArkG1,
    b: ArkG1,
    result: ArkG1,
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for DeferredG1AddWitness {
    fn result(&self) -> Option<&ArkG1> {
        Some(&self.ark_result)
    }
}

#[derive(Clone, Debug)]
struct DeferredG2AddWitness {
    a: ArkG2,
    b: ArkG2,
    result: ArkG2,
    ark_result: ArkG2,
}

impl WitnessResult<ArkG2> for DeferredG2AddWitness {
    fn result(&self) -> Option<&ArkG2> {
        Some(&self.ark_result)
    }
}

impl WitnessBackend for DeferredJoltWitness {
    type G1AddWitness = DeferredG1AddWitness;
    type G1ScalarMulWitness = DeferredG1ScalarMulWitness;
    type MsmG1Witness = UnimplementedWitness<ArkG1>;

    type G2AddWitness = DeferredG2AddWitness;
    type G2ScalarMulWitness = DeferredG2ScalarMulWitness;
    type MsmG2Witness = UnimplementedWitness<ArkG2>;

    type GtMulWitness = DeferredGtMulWitness;
    type GtExpWitness = DeferredGtExpWitness;

    #[cfg(feature = "experimental-pairing-recursion")]
    type PairingWitness = JoltMultiMillerLoopWitness;
    #[cfg(not(feature = "experimental-pairing-recursion"))]
    type PairingWitness = UnimplementedWitness<ArkGT>;

    #[cfg(feature = "experimental-pairing-recursion")]
    type MultiPairingWitness = JoltMultiMillerLoopWitness;
    #[cfg(not(feature = "experimental-pairing-recursion"))]
    type MultiPairingWitness = UnimplementedWitness<ArkGT>;
}

struct DeferredJoltWitnessGenerator;

impl WitnessGenerator<DeferredJoltWitness, BN254> for DeferredJoltWitnessGenerator {
    fn generate_g1_add(
        a: &<BN254 as PairingCurve>::G1,
        b: &<BN254 as PairingCurve>::G1,
        result: &<BN254 as PairingCurve>::G1,
    ) -> DeferredG1AddWitness {
        DeferredG1AddWitness {
            a: *a,
            b: *b,
            result: *result,
            ark_result: *result,
        }
    }

    fn generate_g2_add(
        a: &<BN254 as PairingCurve>::G2,
        b: &<BN254 as PairingCurve>::G2,
        result: &<BN254 as PairingCurve>::G2,
    ) -> DeferredG2AddWitness {
        DeferredG2AddWitness {
            a: *a,
            b: *b,
            result: *result,
            ark_result: *result,
        }
    }

    fn generate_gt_exp(
        base: &<BN254 as PairingCurve>::GT,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::GT,
    ) -> DeferredGtExpWitness {
        let scalar_fr = ark_to_jolt(scalar);
        DeferredGtExpWitness {
            base: base.0,
            exponent: scalar_fr,
            result: result.0,
            ark_result: *result,
        }
    }

    fn generate_gt_mul(
        lhs: &<BN254 as PairingCurve>::GT,
        rhs: &<BN254 as PairingCurve>::GT,
        result: &<BN254 as PairingCurve>::GT,
    ) -> DeferredGtMulWitness {
        DeferredGtMulWitness {
            lhs: lhs.0,
            rhs: rhs.0,
            result: result.0,
            ark_result: *result,
        }
    }

    fn generate_g1_scalar_mul(
        point: &<BN254 as PairingCurve>::G1,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::G1,
    ) -> DeferredG1ScalarMulWitness {
        let point_affine: G1Affine = point.0.into();
        let scalar_fr = ark_to_jolt(scalar);
        let result_affine: G1Affine = result.0.into();
        DeferredG1ScalarMulWitness {
            point_base: point_affine,
            scalar: scalar_fr,
            result: result_affine,
            ark_result: *result,
        }
    }

    fn generate_g2_scalar_mul(
        point: &<BN254 as PairingCurve>::G2,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::G2,
    ) -> DeferredG2ScalarMulWitness {
        let point_affine: G2Affine = point.0.into();
        let scalar_fr = ark_to_jolt(scalar);
        let result_affine: G2Affine = result.0.into();
        DeferredG2ScalarMulWitness {
            point_base: point_affine,
            scalar: scalar_fr,
            result: result_affine,
            ark_result: *result,
        }
    }

    fn generate_pairing(
        g1: &<BN254 as PairingCurve>::G1,
        g2: &<BN254 as PairingCurve>::G2,
        result: &<BN254 as PairingCurve>::GT,
    ) -> <DeferredJoltWitness as WitnessBackend>::PairingWitness {
        #[cfg(feature = "experimental-pairing-recursion")]
        {
            // Keep the same behavior as `JoltWitnessGenerator` when pairing recursion is enabled.
            let g1_affine: G1Affine = g1.0.into();
            let g2_affine: G2Affine = g2.0.into();
            let steps = MultiMillerLoopSteps::new(&[g1_affine], &[g2_affine]);
            JoltMultiMillerLoopWitness {
                f_packed_mles: steps.f_packed_mles,
                f_next_packed_mles: steps.f_next_packed_mles,
                quotient_packed_mles: steps.quotient_packed_mles,
                t_x_c0_packed_mles: steps.t_x_c0_packed_mles,
                t_x_c1_packed_mles: steps.t_x_c1_packed_mles,
                t_y_c0_packed_mles: steps.t_y_c0_packed_mles,
                t_y_c1_packed_mles: steps.t_y_c1_packed_mles,
                t_x_c0_next_packed_mles: steps.t_x_c0_next_packed_mles,
                t_x_c1_next_packed_mles: steps.t_x_c1_next_packed_mles,
                t_y_c0_next_packed_mles: steps.t_y_c0_next_packed_mles,
                t_y_c1_next_packed_mles: steps.t_y_c1_next_packed_mles,
                lambda_c0_packed_mles: steps.lambda_c0_packed_mles,
                lambda_c1_packed_mles: steps.lambda_c1_packed_mles,
                inv_dx_c0_packed_mles: steps.inv_dx_c0_packed_mles,
                inv_dx_c1_packed_mles: steps.inv_dx_c1_packed_mles,
                inv_two_y_c0_packed_mles: steps.inv_two_y_c0_packed_mles,
                inv_two_y_c1_packed_mles: steps.inv_two_y_c1_packed_mles,
                x_p_packed_mles: steps.x_p_packed_mles,
                y_p_packed_mles: steps.y_p_packed_mles,
                x_q_c0_packed_mles: steps.x_q_c0_packed_mles,
                x_q_c1_packed_mles: steps.x_q_c1_packed_mles,
                y_q_c0_packed_mles: steps.y_q_c0_packed_mles,
                y_q_c1_packed_mles: steps.y_q_c1_packed_mles,
                is_double_packed_mles: steps.is_double_packed_mles,
                is_add_packed_mles: steps.is_add_packed_mles,
                l_val_packed_mles: steps.l_val_packed_mles,
                num_steps: steps.num_steps,
                ark_result: *result,
            }
        }
        #[cfg(not(feature = "experimental-pairing-recursion"))]
        {
            let _ = (g1, g2, result);
            UnimplementedWitness::new("Pairing (disabled: experimental-pairing-recursion)")
        }
    }

    fn generate_multi_pairing(
        g1s: &[<BN254 as PairingCurve>::G1],
        g2s: &[<BN254 as PairingCurve>::G2],
        result: &<BN254 as PairingCurve>::GT,
    ) -> <DeferredJoltWitness as WitnessBackend>::MultiPairingWitness {
        #[cfg(feature = "experimental-pairing-recursion")]
        {
            let g1_affines: Vec<G1Affine> = g1s.iter().map(|p| p.0.into()).collect();
            let g2_affines: Vec<G2Affine> = g2s.iter().map(|p| p.0.into()).collect();
            let steps = MultiMillerLoopSteps::new(&g1_affines, &g2_affines);
            JoltMultiMillerLoopWitness {
                f_packed_mles: steps.f_packed_mles,
                f_next_packed_mles: steps.f_next_packed_mles,
                quotient_packed_mles: steps.quotient_packed_mles,
                t_x_c0_packed_mles: steps.t_x_c0_packed_mles,
                t_x_c1_packed_mles: steps.t_x_c1_packed_mles,
                t_y_c0_packed_mles: steps.t_y_c0_packed_mles,
                t_y_c1_packed_mles: steps.t_y_c1_packed_mles,
                t_x_c0_next_packed_mles: steps.t_x_c0_next_packed_mles,
                t_x_c1_next_packed_mles: steps.t_x_c1_next_packed_mles,
                t_y_c0_next_packed_mles: steps.t_y_c0_next_packed_mles,
                t_y_c1_next_packed_mles: steps.t_y_c1_next_packed_mles,
                lambda_c0_packed_mles: steps.lambda_c0_packed_mles,
                lambda_c1_packed_mles: steps.lambda_c1_packed_mles,
                inv_dx_c0_packed_mles: steps.inv_dx_c0_packed_mles,
                inv_dx_c1_packed_mles: steps.inv_dx_c1_packed_mles,
                inv_two_y_c0_packed_mles: steps.inv_two_y_c0_packed_mles,
                inv_two_y_c1_packed_mles: steps.inv_two_y_c1_packed_mles,
                x_p_packed_mles: steps.x_p_packed_mles,
                y_p_packed_mles: steps.y_p_packed_mles,
                x_q_c0_packed_mles: steps.x_q_c0_packed_mles,
                x_q_c1_packed_mles: steps.x_q_c1_packed_mles,
                y_q_c0_packed_mles: steps.y_q_c0_packed_mles,
                y_q_c1_packed_mles: steps.y_q_c1_packed_mles,
                is_double_packed_mles: steps.is_double_packed_mles,
                is_add_packed_mles: steps.is_add_packed_mles,
                l_val_packed_mles: steps.l_val_packed_mles,
                num_steps: steps.num_steps,
                ark_result: *result,
            }
        }
        #[cfg(not(feature = "experimental-pairing-recursion"))]
        {
            let _ = (g1s, g2s, result);
            UnimplementedWitness::new("Multi-pairing (disabled: experimental-pairing-recursion)")
        }
    }

    fn generate_msm_g1(
        _bases: &[<BN254 as PairingCurve>::G1],
        _scalars: &[<<BN254 as PairingCurve>::G1 as Group>::Scalar],
        _result: &<BN254 as PairingCurve>::G1,
    ) -> UnimplementedWitness<ArkG1> {
        UnimplementedWitness::new("G1 MSM")
    }

    fn generate_msm_g2(
        _bases: &[<BN254 as PairingCurve>::G2],
        _scalars: &[<<BN254 as PairingCurve>::G1 as Group>::Scalar],
        _result: &<BN254 as PairingCurve>::G2,
    ) -> UnimplementedWitness<ArkG2> {
        UnimplementedWitness::new("G2 MSM")
    }
}

fn expand_deferred_witnesses(
    deferred: WitnessCollection<DeferredJoltWitness>,
) -> WitnessCollection<JoltWitness> {
    let mut out = WitnessCollection::<JoltWitness>::new();
    out.num_rounds = deferred.num_rounds;

    // Parallel expansion on host; deterministic ordering is enforced later by OpId sorting.
    #[cfg(not(any(target_arch = "riscv64", target_arch = "riscv32")))]
    {
        use rayon::prelude::*;

        out.gt_exp = deferred
            .gt_exp
            .into_par_iter()
            .map(|(op_id, w)| {
                let exp_steps = Base4ExponentiationSteps::new(w.base, w.exponent);
                debug_assert_eq!(exp_steps.result, w.result);
                (
                    op_id,
                    JoltGtExpWitness {
                        base: exp_steps.base,
                        exponent: exp_steps.exponent,
                        result: exp_steps.result,
                        rho_mles: exp_steps.rho_mles,
                        quotient_mles: exp_steps.quotient_mles,
                        bits: exp_steps.bits,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        out.gt_mul = deferred
            .gt_mul
            .into_par_iter()
            .map(|(op_id, w)| {
                let mul_steps = MultiplicationSteps::new(w.lhs, w.rhs);
                debug_assert_eq!(mul_steps.result, w.result);
                (
                    op_id,
                    JoltGtMulWitness {
                        lhs: mul_steps.lhs,
                        rhs: mul_steps.rhs,
                        result: mul_steps.result,
                        quotient_mle: mul_steps.quotient_mle,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        out.g1_scalar_mul = deferred
            .g1_scalar_mul
            .into_par_iter()
            .map(|(op_id, w)| {
                let steps = ScalarMultiplicationSteps::new(w.point_base, w.scalar);
                debug_assert_eq!(steps.result, w.result);
                (
                    op_id,
                    JoltG1ScalarMulWitness {
                        point_base: steps.point_base,
                        scalar: steps.scalar,
                        result: steps.result,
                        x_a_mles: steps.x_a_mles,
                        y_a_mles: steps.y_a_mles,
                        x_t_mles: steps.x_t_mles,
                        y_t_mles: steps.y_t_mles,
                        x_a_next_mles: steps.x_a_next_mles,
                        y_a_next_mles: steps.y_a_next_mles,
                        t_is_infinity_mles: steps.t_is_infinity_mles,
                        a_is_infinity_mles: steps.a_is_infinity_mles,
                        bit_mles: steps.bit_mles,
                        bits: steps.bits,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        out.g2_scalar_mul = deferred
            .g2_scalar_mul
            .into_par_iter()
            .map(|(op_id, w)| {
                let steps = G2ScalarMultiplicationSteps::new(w.point_base, w.scalar);
                debug_assert_eq!(steps.result, w.result);
                (
                    op_id,
                    JoltG2ScalarMulWitness {
                        point_base: steps.point_base,
                        scalar: steps.scalar,
                        result: steps.result,
                        x_a_c0_mles: steps.x_a_c0_mles,
                        x_a_c1_mles: steps.x_a_c1_mles,
                        y_a_c0_mles: steps.y_a_c0_mles,
                        y_a_c1_mles: steps.y_a_c1_mles,
                        x_t_c0_mles: steps.x_t_c0_mles,
                        x_t_c1_mles: steps.x_t_c1_mles,
                        y_t_c0_mles: steps.y_t_c0_mles,
                        y_t_c1_mles: steps.y_t_c1_mles,
                        x_a_next_c0_mles: steps.x_a_next_c0_mles,
                        x_a_next_c1_mles: steps.x_a_next_c1_mles,
                        y_a_next_c0_mles: steps.y_a_next_c0_mles,
                        y_a_next_c1_mles: steps.y_a_next_c1_mles,
                        t_is_infinity_mles: steps.t_is_infinity_mles,
                        a_is_infinity_mles: steps.a_is_infinity_mles,
                        bit_mles: steps.bit_mles,
                        bits: steps.bits,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        // Adds are cheap; expand sequentially to avoid overhead.
        out.g1_add = deferred
            .g1_add
            .into_iter()
            .map(|(op_id, w)| (op_id, G1AdditionSteps::new(&w.a, &w.b, &w.result)))
            .collect();
        out.g2_add = deferred
            .g2_add
            .into_iter()
            .map(|(op_id, w)| (op_id, G2AdditionSteps::new(&w.a, &w.b, &w.result)))
            .collect();
        out.msm_g1 = deferred.msm_g1;
        out.msm_g2 = deferred.msm_g2;
        out.pairing = deferred.pairing;
        out.multi_pairing = deferred.multi_pairing;
    }

    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    {
        out.gt_exp = deferred
            .gt_exp
            .into_iter()
            .map(|(op_id, w)| {
                let exp_steps = Base4ExponentiationSteps::new(w.base, w.exponent);
                (
                    op_id,
                    JoltGtExpWitness {
                        base: exp_steps.base,
                        exponent: exp_steps.exponent,
                        result: exp_steps.result,
                        rho_mles: exp_steps.rho_mles,
                        quotient_mles: exp_steps.quotient_mles,
                        bits: exp_steps.bits,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect();
        out.gt_mul = deferred
            .gt_mul
            .into_iter()
            .map(|(op_id, w)| {
                let mul_steps = MultiplicationSteps::new(w.lhs, w.rhs);
                (
                    op_id,
                    JoltGtMulWitness {
                        lhs: mul_steps.lhs,
                        rhs: mul_steps.rhs,
                        result: mul_steps.result,
                        quotient_mle: mul_steps.quotient_mle,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect();
        out.g1_scalar_mul = deferred
            .g1_scalar_mul
            .into_iter()
            .map(|(op_id, w)| {
                let steps = ScalarMultiplicationSteps::new(w.point_base, w.scalar);
                (
                    op_id,
                    JoltG1ScalarMulWitness {
                        point_base: steps.point_base,
                        scalar: steps.scalar,
                        result: steps.result,
                        x_a_mles: steps.x_a_mles,
                        y_a_mles: steps.y_a_mles,
                        x_t_mles: steps.x_t_mles,
                        y_t_mles: steps.y_t_mles,
                        x_a_next_mles: steps.x_a_next_mles,
                        y_a_next_mles: steps.y_a_next_mles,
                        t_is_infinity_mles: steps.t_is_infinity_mles,
                        a_is_infinity_mles: steps.a_is_infinity_mles,
                        bit_mles: steps.bit_mles,
                        bits: steps.bits,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect();
        out.g2_scalar_mul = deferred
            .g2_scalar_mul
            .into_iter()
            .map(|(op_id, w)| {
                let steps = G2ScalarMultiplicationSteps::new(w.point_base, w.scalar);
                (
                    op_id,
                    JoltG2ScalarMulWitness {
                        point_base: steps.point_base,
                        scalar: steps.scalar,
                        result: steps.result,
                        x_a_c0_mles: steps.x_a_c0_mles,
                        x_a_c1_mles: steps.x_a_c1_mles,
                        y_a_c0_mles: steps.y_a_c0_mles,
                        y_a_c1_mles: steps.y_a_c1_mles,
                        x_t_c0_mles: steps.x_t_c0_mles,
                        x_t_c1_mles: steps.x_t_c1_mles,
                        y_t_c0_mles: steps.y_t_c0_mles,
                        y_t_c1_mles: steps.y_t_c1_mles,
                        x_a_next_c0_mles: steps.x_a_next_c0_mles,
                        x_a_next_c1_mles: steps.x_a_next_c1_mles,
                        y_a_next_c0_mles: steps.y_a_next_c0_mles,
                        y_a_next_c1_mles: steps.y_a_next_c1_mles,
                        t_is_infinity_mles: steps.t_is_infinity_mles,
                        a_is_infinity_mles: steps.a_is_infinity_mles,
                        bit_mles: steps.bit_mles,
                        bits: steps.bits,
                        ark_result: w.ark_result,
                    },
                )
            })
            .collect();
        out.g1_add = deferred
            .g1_add
            .into_iter()
            .map(|(op_id, w)| (op_id, G1AdditionSteps::new(&w.a, &w.b, &w.result)))
            .collect();
        out.g2_add = deferred
            .g2_add
            .into_iter()
            .map(|(op_id, w)| (op_id, G2AdditionSteps::new(&w.a, &w.b, &w.result)))
            .collect();
        out.msm_g1 = deferred.msm_g1;
        out.msm_g2 = deferred.msm_g2;
        out.pairing = deferred.pairing;
        out.multi_pairing = deferred.multi_pairing;
    }

    out
}

// NOTE: Legacy hint-based and backend-specific AST construction code has been removed.
// Upstream Dory now provides `TraceContext::{for_witness_gen_with_ast, for_symbolic}` which we
// use to generate both witnesses and the symbolic AST.

/// GTExp witness following the ExponentiationSteps pattern
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGtExpWitness {
    pub base: Fq12,
    pub exponent: Fr,
    pub result: Fq12,
    pub rho_mles: Vec<Vec<Fq>>,
    pub quotient_mles: Vec<Vec<Fq>>,
    pub bits: Vec<bool>,
    ark_result: ArkGT,
}

impl WitnessResult<ArkGT> for JoltGtExpWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

/// GTMul witness following the MultiplicationSteps pattern
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGtMulWitness {
    pub lhs: Fq12,             // Left operand (a)
    pub rhs: Fq12,             // Right operand (b)
    pub result: Fq12,          // Product (c = a × b)
    pub quotient_mle: Vec<Fq>, // Quotient polynomial Q(x)
    ark_result: ArkGT,         // For WitnessResult trait
}

impl WitnessResult<ArkGT> for JoltGtMulWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

/// G1 scalar multiplication witness following the ScalarMultiplicationSteps pattern
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltG1ScalarMulWitness {
    pub point_base: G1Affine,
    pub scalar: Fr,
    pub result: G1Affine,
    pub x_a_mles: Vec<Vec<Fq>>,           // x-coords of A_0, A_1, ..., A_n
    pub y_a_mles: Vec<Vec<Fq>>,           // y-coords of A_0, A_1, ..., A_n
    pub x_t_mles: Vec<Vec<Fq>>,           // x-coords of T_0, T_1, ..., T_{n-1}
    pub y_t_mles: Vec<Vec<Fq>>,           // y-coords of T_0, T_1, ..., T_{n-1}
    pub x_a_next_mles: Vec<Vec<Fq>>,      // x-coords of A_{i+1} (shifted by 1)
    pub y_a_next_mles: Vec<Vec<Fq>>,      // y-coords of A_{i+1} (shifted by 1)
    pub t_is_infinity_mles: Vec<Vec<Fq>>, // 1 if T_i = O, 0 otherwise
    pub a_is_infinity_mles: Vec<Vec<Fq>>, // 1 if A_i = O, 0 otherwise
    pub bit_mles: Vec<Vec<Fq>>,           // Scalar bit b_i (CRITICAL for soundness)
    pub bits: Vec<bool>,
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for JoltG1ScalarMulWitness {
    fn result(&self) -> Option<&ArkG1> {
        Some(&self.ark_result)
    }
}

/// G2 scalar multiplication witness following the ScalarMultiplicationSteps pattern,
/// with Fq2 coordinates split into (c0, c1) components in Fq.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltG2ScalarMulWitness {
    pub point_base: G2Affine,
    pub scalar: Fr,
    pub result: G2Affine,

    pub x_a_c0_mles: Vec<Vec<Fq>>,
    pub x_a_c1_mles: Vec<Vec<Fq>>,
    pub y_a_c0_mles: Vec<Vec<Fq>>,
    pub y_a_c1_mles: Vec<Vec<Fq>>,

    pub x_t_c0_mles: Vec<Vec<Fq>>,
    pub x_t_c1_mles: Vec<Vec<Fq>>,
    pub y_t_c0_mles: Vec<Vec<Fq>>,
    pub y_t_c1_mles: Vec<Vec<Fq>>,

    pub x_a_next_c0_mles: Vec<Vec<Fq>>,
    pub x_a_next_c1_mles: Vec<Vec<Fq>>,
    pub y_a_next_c0_mles: Vec<Vec<Fq>>,
    pub y_a_next_c1_mles: Vec<Vec<Fq>>,

    pub t_is_infinity_mles: Vec<Vec<Fq>>,
    pub a_is_infinity_mles: Vec<Vec<Fq>>,
    pub bit_mles: Vec<Vec<Fq>>,
    pub bits: Vec<bool>,
    ark_result: ArkG2,
}

impl WitnessResult<ArkG2> for JoltG2ScalarMulWitness {
    fn result(&self) -> Option<&ArkG2> {
        Some(&self.ark_result)
    }
}

/// Multi-Miller loop witness for Dory recursion.
#[cfg(feature = "experimental-pairing-recursion")]
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltMultiMillerLoopWitness {
    pub f_packed_mles: Vec<Vec<Fq>>,
    pub f_next_packed_mles: Vec<Vec<Fq>>,
    pub quotient_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c0_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c1_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c0_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c1_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c0_next_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c1_next_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c0_next_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c1_next_packed_mles: Vec<Vec<Fq>>,
    pub lambda_c0_packed_mles: Vec<Vec<Fq>>,
    pub lambda_c1_packed_mles: Vec<Vec<Fq>>,
    pub inv_dx_c0_packed_mles: Vec<Vec<Fq>>,
    pub inv_dx_c1_packed_mles: Vec<Vec<Fq>>,
    pub inv_two_y_c0_packed_mles: Vec<Vec<Fq>>,
    pub inv_two_y_c1_packed_mles: Vec<Vec<Fq>>,
    pub x_p_packed_mles: Vec<Vec<Fq>>,
    pub y_p_packed_mles: Vec<Vec<Fq>>,
    pub x_q_c0_packed_mles: Vec<Vec<Fq>>,
    pub x_q_c1_packed_mles: Vec<Vec<Fq>>,
    pub y_q_c0_packed_mles: Vec<Vec<Fq>>,
    pub y_q_c1_packed_mles: Vec<Vec<Fq>>,
    pub is_double_packed_mles: Vec<Vec<Fq>>,
    pub is_add_packed_mles: Vec<Vec<Fq>>,
    pub l_val_packed_mles: Vec<Vec<Fq>>,
    pub num_steps: usize,
    ark_result: ArkGT,
}

#[cfg(feature = "experimental-pairing-recursion")]
impl WitnessResult<ArkGT> for JoltMultiMillerLoopWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

/// Witness type for unimplemented operations that panics when used
#[derive(Clone, Debug)]
pub struct UnimplementedWitness<T> {
    _operation: &'static str,
    _marker: PhantomData<T>,
}

impl<T> UnimplementedWitness<T> {
    fn new(operation: &'static str) -> Self {
        Self {
            _operation: operation,
            _marker: PhantomData,
        }
    }
}

impl WitnessResult<ArkG1> for UnimplementedWitness<ArkG1> {
    fn result(&self) -> Option<&ArkG1> {
        None
    }
}

impl WitnessResult<ArkG2> for UnimplementedWitness<ArkG2> {
    fn result(&self) -> Option<&ArkG2> {
        None
    }
}

impl WitnessResult<ArkGT> for UnimplementedWitness<ArkGT> {
    fn result(&self) -> Option<&ArkGT> {
        None
    }
}

impl WitnessBackend for JoltWitness {
    // G1 operations
    type G1AddWitness = G1AdditionSteps;
    type G1ScalarMulWitness = JoltG1ScalarMulWitness;
    type MsmG1Witness = UnimplementedWitness<ArkG1>;

    // G2 operations
    type G2AddWitness = G2AdditionSteps;
    type G2ScalarMulWitness = JoltG2ScalarMulWitness;
    type MsmG2Witness = UnimplementedWitness<ArkG2>;

    // GT operations
    type GtMulWitness = JoltGtMulWitness;
    type GtExpWitness = JoltGtExpWitness;

    // Pairing operations
    #[cfg(feature = "experimental-pairing-recursion")]
    type PairingWitness = JoltMultiMillerLoopWitness;
    #[cfg(not(feature = "experimental-pairing-recursion"))]
    type PairingWitness = UnimplementedWitness<ArkGT>;

    #[cfg(feature = "experimental-pairing-recursion")]
    type MultiPairingWitness = JoltMultiMillerLoopWitness;
    #[cfg(not(feature = "experimental-pairing-recursion"))]
    type MultiPairingWitness = UnimplementedWitness<ArkGT>;
}

pub struct JoltWitnessGenerator;

impl WitnessGenerator<JoltWitness, BN254> for JoltWitnessGenerator {
    fn generate_g1_add(
        a: &<BN254 as PairingCurve>::G1,
        b: &<BN254 as PairingCurve>::G1,
        result: &<BN254 as PairingCurve>::G1,
    ) -> G1AdditionSteps {
        G1AdditionSteps::new(a, b, result)
    }

    fn generate_g2_add(
        a: &<BN254 as PairingCurve>::G2,
        b: &<BN254 as PairingCurve>::G2,
        result: &<BN254 as PairingCurve>::G2,
    ) -> G2AdditionSteps {
        G2AdditionSteps::new(a, b, result)
    }

    fn generate_gt_exp(
        base: &<BN254 as PairingCurve>::GT,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltGtExpWitness {
        let base_fq12 = base.0;
        let scalar_fr = ark_to_jolt(scalar);

        let exp_steps = Base4ExponentiationSteps::new(base_fq12, scalar_fr);

        debug_assert_eq!(
            exp_steps.result, result.0,
            "ExponentiationSteps result doesn't match expected result"
        );

        JoltGtExpWitness {
            base: exp_steps.base,
            exponent: exp_steps.exponent,
            result: exp_steps.result,
            rho_mles: exp_steps.rho_mles,
            quotient_mles: exp_steps.quotient_mles,
            bits: exp_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_g1_scalar_mul(
        point: &<BN254 as PairingCurve>::G1,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::G1,
    ) -> JoltG1ScalarMulWitness {
        let point_affine: G1Affine = point.0.into();
        let scalar_fr = ark_to_jolt(scalar);

        let scalar_mul_steps = ScalarMultiplicationSteps::new(point_affine, scalar_fr);

        let result_affine: G1Affine = result.0.into();
        debug_assert_eq!(
            scalar_mul_steps.result, result_affine,
            "ScalarMultiplicationSteps result doesn't match expected result"
        );

        JoltG1ScalarMulWitness {
            point_base: scalar_mul_steps.point_base,
            scalar: scalar_mul_steps.scalar,
            result: scalar_mul_steps.result,
            x_a_mles: scalar_mul_steps.x_a_mles,
            y_a_mles: scalar_mul_steps.y_a_mles,
            x_t_mles: scalar_mul_steps.x_t_mles,
            y_t_mles: scalar_mul_steps.y_t_mles,
            x_a_next_mles: scalar_mul_steps.x_a_next_mles,
            y_a_next_mles: scalar_mul_steps.y_a_next_mles,
            t_is_infinity_mles: scalar_mul_steps.t_is_infinity_mles,
            a_is_infinity_mles: scalar_mul_steps.a_is_infinity_mles,
            bit_mles: scalar_mul_steps.bit_mles,
            bits: scalar_mul_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_g2_scalar_mul(
        point: &<BN254 as PairingCurve>::G2,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::G2,
    ) -> JoltG2ScalarMulWitness {
        let point_affine: G2Affine = point.0.into();
        let scalar_fr = ark_to_jolt(scalar);

        let scalar_mul_steps = G2ScalarMultiplicationSteps::new(point_affine, scalar_fr);

        let result_affine: G2Affine = result.0.into();
        debug_assert_eq!(
            scalar_mul_steps.result, result_affine,
            "G2ScalarMultiplicationSteps result doesn't match expected result"
        );

        JoltG2ScalarMulWitness {
            point_base: scalar_mul_steps.point_base,
            scalar: scalar_mul_steps.scalar,
            result: scalar_mul_steps.result,
            x_a_c0_mles: scalar_mul_steps.x_a_c0_mles,
            x_a_c1_mles: scalar_mul_steps.x_a_c1_mles,
            y_a_c0_mles: scalar_mul_steps.y_a_c0_mles,
            y_a_c1_mles: scalar_mul_steps.y_a_c1_mles,
            x_t_c0_mles: scalar_mul_steps.x_t_c0_mles,
            x_t_c1_mles: scalar_mul_steps.x_t_c1_mles,
            y_t_c0_mles: scalar_mul_steps.y_t_c0_mles,
            y_t_c1_mles: scalar_mul_steps.y_t_c1_mles,
            x_a_next_c0_mles: scalar_mul_steps.x_a_next_c0_mles,
            x_a_next_c1_mles: scalar_mul_steps.x_a_next_c1_mles,
            y_a_next_c0_mles: scalar_mul_steps.y_a_next_c0_mles,
            y_a_next_c1_mles: scalar_mul_steps.y_a_next_c1_mles,
            t_is_infinity_mles: scalar_mul_steps.t_is_infinity_mles,
            a_is_infinity_mles: scalar_mul_steps.a_is_infinity_mles,
            bit_mles: scalar_mul_steps.bit_mles,
            bits: scalar_mul_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_gt_mul(
        lhs: &<BN254 as PairingCurve>::GT,
        rhs: &<BN254 as PairingCurve>::GT,
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltGtMulWitness {
        let lhs_fq12 = lhs.0;
        let rhs_fq12 = rhs.0;

        let mul_steps = MultiplicationSteps::new(lhs_fq12, rhs_fq12);

        debug_assert_eq!(
            mul_steps.result, result.0,
            "MultiplicationSteps result doesn't match expected result"
        );

        JoltGtMulWitness {
            lhs: mul_steps.lhs,
            rhs: mul_steps.rhs,
            result: mul_steps.result,
            quotient_mle: mul_steps.quotient_mle,
            ark_result: *result,
        }
    }

    fn generate_pairing(
        g1: &<BN254 as PairingCurve>::G1,
        g2: &<BN254 as PairingCurve>::G2,
        result: &<BN254 as PairingCurve>::GT,
    ) -> <JoltWitness as WitnessBackend>::PairingWitness {
        #[cfg(feature = "experimental-pairing-recursion")]
        {
            let g1_affine: G1Affine = g1.0.into();
            let g2_affine: G2Affine = g2.0.into();

            let steps = MultiMillerLoopSteps::new(&[g1_affine], &[g2_affine]);

            JoltMultiMillerLoopWitness {
                f_packed_mles: steps.f_packed_mles,
                f_next_packed_mles: steps.f_next_packed_mles,
                quotient_packed_mles: steps.quotient_packed_mles,
                t_x_c0_packed_mles: steps.t_x_c0_packed_mles,
                t_x_c1_packed_mles: steps.t_x_c1_packed_mles,
                t_y_c0_packed_mles: steps.t_y_c0_packed_mles,
                t_y_c1_packed_mles: steps.t_y_c1_packed_mles,
                t_x_c0_next_packed_mles: steps.t_x_c0_next_packed_mles,
                t_x_c1_next_packed_mles: steps.t_x_c1_next_packed_mles,
                t_y_c0_next_packed_mles: steps.t_y_c0_next_packed_mles,
                t_y_c1_next_packed_mles: steps.t_y_c1_next_packed_mles,
                lambda_c0_packed_mles: steps.lambda_c0_packed_mles,
                lambda_c1_packed_mles: steps.lambda_c1_packed_mles,
                inv_dx_c0_packed_mles: steps.inv_dx_c0_packed_mles,
                inv_dx_c1_packed_mles: steps.inv_dx_c1_packed_mles,
                inv_two_y_c0_packed_mles: steps.inv_two_y_c0_packed_mles,
                inv_two_y_c1_packed_mles: steps.inv_two_y_c1_packed_mles,
                x_p_packed_mles: steps.x_p_packed_mles,
                y_p_packed_mles: steps.y_p_packed_mles,
                x_q_c0_packed_mles: steps.x_q_c0_packed_mles,
                x_q_c1_packed_mles: steps.x_q_c1_packed_mles,
                y_q_c0_packed_mles: steps.y_q_c0_packed_mles,
                y_q_c1_packed_mles: steps.y_q_c1_packed_mles,
                is_double_packed_mles: steps.is_double_packed_mles,
                is_add_packed_mles: steps.is_add_packed_mles,
                l_val_packed_mles: steps.l_val_packed_mles,
                num_steps: steps.num_steps,
                ark_result: *result,
            }
        }
        #[cfg(not(feature = "experimental-pairing-recursion"))]
        {
            let _ = (g1, g2, result);
            UnimplementedWitness::new("Pairing (disabled: experimental-pairing-recursion)")
        }
    }

    fn generate_multi_pairing(
        g1s: &[<BN254 as PairingCurve>::G1],
        g2s: &[<BN254 as PairingCurve>::G2],
        result: &<BN254 as PairingCurve>::GT,
    ) -> <JoltWitness as WitnessBackend>::MultiPairingWitness {
        #[cfg(feature = "experimental-pairing-recursion")]
        {
            let g1_affines: Vec<G1Affine> = g1s.iter().map(|p| p.0.into()).collect();
            let g2_affines: Vec<G2Affine> = g2s.iter().map(|p| p.0.into()).collect();

            let steps = MultiMillerLoopSteps::new(&g1_affines, &g2_affines);

            JoltMultiMillerLoopWitness {
                f_packed_mles: steps.f_packed_mles,
                f_next_packed_mles: steps.f_next_packed_mles,
                quotient_packed_mles: steps.quotient_packed_mles,
                t_x_c0_packed_mles: steps.t_x_c0_packed_mles,
                t_x_c1_packed_mles: steps.t_x_c1_packed_mles,
                t_y_c0_packed_mles: steps.t_y_c0_packed_mles,
                t_y_c1_packed_mles: steps.t_y_c1_packed_mles,
                t_x_c0_next_packed_mles: steps.t_x_c0_next_packed_mles,
                t_x_c1_next_packed_mles: steps.t_x_c1_next_packed_mles,
                t_y_c0_next_packed_mles: steps.t_y_c0_next_packed_mles,
                t_y_c1_next_packed_mles: steps.t_y_c1_next_packed_mles,
                lambda_c0_packed_mles: steps.lambda_c0_packed_mles,
                lambda_c1_packed_mles: steps.lambda_c1_packed_mles,
                inv_dx_c0_packed_mles: steps.inv_dx_c0_packed_mles,
                inv_dx_c1_packed_mles: steps.inv_dx_c1_packed_mles,
                inv_two_y_c0_packed_mles: steps.inv_two_y_c0_packed_mles,
                inv_two_y_c1_packed_mles: steps.inv_two_y_c1_packed_mles,
                x_p_packed_mles: steps.x_p_packed_mles,
                y_p_packed_mles: steps.y_p_packed_mles,
                x_q_c0_packed_mles: steps.x_q_c0_packed_mles,
                x_q_c1_packed_mles: steps.x_q_c1_packed_mles,
                y_q_c0_packed_mles: steps.y_q_c0_packed_mles,
                y_q_c1_packed_mles: steps.y_q_c1_packed_mles,
                is_double_packed_mles: steps.is_double_packed_mles,
                is_add_packed_mles: steps.is_add_packed_mles,
                l_val_packed_mles: steps.l_val_packed_mles,
                num_steps: steps.num_steps,
                ark_result: *result,
            }
        }
        #[cfg(not(feature = "experimental-pairing-recursion"))]
        {
            let _ = (g1s, g2s, result);
            UnimplementedWitness::new("Multi-pairing (disabled: experimental-pairing-recursion)")
        }
    }

    fn generate_msm_g1(
        _bases: &[<BN254 as PairingCurve>::G1],
        _scalars: &[<<BN254 as PairingCurve>::G1 as Group>::Scalar],
        _result: &<BN254 as PairingCurve>::G1,
    ) -> UnimplementedWitness<ArkG1> {
        UnimplementedWitness::new("G1 MSM")
    }

    fn generate_msm_g2(
        _bases: &[<BN254 as PairingCurve>::G2],
        _scalars: &[<<BN254 as PairingCurve>::G1 as Group>::Scalar],
        _result: &<BN254 as PairingCurve>::G2,
    ) -> UnimplementedWitness<ArkG2> {
        UnimplementedWitness::new("G2 MSM")
    }
}

/// Expand full Dory recursion witnesses from the deferred (AST + outputs-only hints) capture.
///
/// This is the "Phase 2" of Dory's two-phase witness generation: expensive, per-op witness
/// payloads are generated in parallel given the already-determined operation DAG and outputs.
#[cfg(any())]
fn expand_witnesses_from_deferred(
    ast: &AstGraph<BN254>,
    hints: &HintMap<BN254>,
    setup: &dory::setup::VerifierSetup<BN254>,
    proof: &ArkDoryProof,
    commitment: ArkGT,
) -> Result<dory::recursion::WitnessCollection<JoltWitness>, ProofVerifyError> {
    use std::collections::HashMap;

    debug_assert!(
        ast.validate().is_ok(),
        "invalid Dory recursion AST produced during deferred witness capture"
    );

    let input_provider = DoryInputProviderWithCommitment::new(setup, proof, commitment);

    // Populate a ValueId -> value table.
    //
    // In deferred mode, we *expect* every traced op to have an OpId and a corresponding
    // output hint. However, to be robust to upstream AST changes (e.g. new un-traced nodes),
    // we fall back to computing missing node values from their dependencies.
    let mut values: Vec<Option<EvalResult<BN254>>> = vec![None; ast.nodes.len()];
    for node in &ast.nodes {
        let out_idx = node.out.0 as usize;
        let hinted = node.op.op_id().and_then(|op_id| match node.out_ty {
            ValueType::G1 => hints.get_g1(op_id).copied().map(EvalResult::G1),
            ValueType::G2 => hints.get_g2(op_id).copied().map(EvalResult::G2),
            ValueType::GT => hints.get_gt(op_id).copied().map(EvalResult::GT),
        });

        let value = match (&node.op, hinted) {
            // Fast path: use the recorded output hint.
            (_, Some(v)) => Some(v),

            // Inputs come from setup/proof/commitment.
            (AstOp::Input { .. }, _) => InputProvider::get_input(&input_provider, node),

            // Fallback path: compute from dependencies.
            (AstOp::G1Add { a, b, .. }, _) => {
                let a = match values[a.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G1(v) => *v,
                    _ => unreachable!("G1Add input must be G1"),
                };
                let b = match values[b.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G1(v) => *v,
                    _ => unreachable!("G1Add input must be G1"),
                };
                Some(EvalResult::G1(a.add(&b)))
            }
            (AstOp::G1ScalarMul { point, scalar, .. }, _) => {
                let p = match values[point.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G1(v) => *v,
                    _ => unreachable!("G1ScalarMul input must be G1"),
                };
                Some(EvalResult::G1(p.scale(&scalar.value)))
            }
            (AstOp::G2Add { a, b, .. }, _) => {
                let a = match values[a.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G2(v) => *v,
                    _ => unreachable!("G2Add input must be G2"),
                };
                let b = match values[b.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G2(v) => *v,
                    _ => unreachable!("G2Add input must be G2"),
                };
                Some(EvalResult::G2(a.add(&b)))
            }
            (AstOp::G2ScalarMul { point, scalar, .. }, _) => {
                let p = match values[point.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G2(v) => *v,
                    _ => unreachable!("G2ScalarMul input must be G2"),
                };
                Some(EvalResult::G2(p.scale(&scalar.value)))
            }
            (AstOp::GTMul { lhs, rhs, .. }, _) => {
                let a = match values[lhs.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::GT(v) => *v,
                    _ => unreachable!("GTMul input must be GT"),
                };
                let b = match values[rhs.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::GT(v) => *v,
                    _ => unreachable!("GTMul input must be GT"),
                };
                Some(EvalResult::GT(a.add(&b)))
            }
            (AstOp::GTExp { base, scalar, .. }, _) => {
                let b = match values[base.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::GT(v) => *v,
                    _ => unreachable!("GTExp input must be GT"),
                };
                Some(EvalResult::GT(b.scale(&scalar.value)))
            }
            (AstOp::Pairing { g1, g2, .. }, _) => {
                let p = match values[g1.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G1(v) => *v,
                    _ => unreachable!("Pairing G1 input must be G1"),
                };
                let q = match values[g2.0 as usize].as_ref().expect("AST topo order") {
                    EvalResult::G2(v) => *v,
                    _ => unreachable!("Pairing G2 input must be G2"),
                };
                Some(EvalResult::GT(BN254::pair(&p, &q)))
            }
            (AstOp::MultiPairing { g1s, g2s, .. }, _) => {
                let ps: Vec<<BN254 as PairingCurve>::G1> = g1s
                    .iter()
                    .map(
                        |id| match values[id.0 as usize].as_ref().expect("AST topo order") {
                            EvalResult::G1(v) => *v,
                            _ => unreachable!("MultiPairing G1 input must be G1"),
                        },
                    )
                    .collect();
                let qs: Vec<<BN254 as PairingCurve>::G2> = g2s
                    .iter()
                    .map(
                        |id| match values[id.0 as usize].as_ref().expect("AST topo order") {
                            EvalResult::G2(v) => *v,
                            _ => unreachable!("MultiPairing G2 input must be G2"),
                        },
                    )
                    .collect();
                Some(EvalResult::GT(BN254::multi_pair(&ps, &qs)))
            }
            (
                AstOp::MsmG1 {
                    points, scalars, ..
                },
                _,
            ) => {
                let mut acc = <BN254 as PairingCurve>::G1::identity();
                for (p, s) in points.iter().zip(scalars.iter()) {
                    let base = match values[p.0 as usize].as_ref().expect("AST topo order") {
                        EvalResult::G1(v) => *v,
                        _ => unreachable!("MsmG1 input must be G1"),
                    };
                    acc = acc.add(&base.scale(&s.value));
                }
                Some(EvalResult::G1(acc))
            }
            (
                AstOp::MsmG2 {
                    points, scalars, ..
                },
                _,
            ) => {
                let mut acc = <BN254 as PairingCurve>::G2::identity();
                for (p, s) in points.iter().zip(scalars.iter()) {
                    let base = match values[p.0 as usize].as_ref().expect("AST topo order") {
                        EvalResult::G2(v) => *v,
                        _ => unreachable!("MsmG2 input must be G2"),
                    };
                    acc = acc.add(&base.scale(&s.value));
                }
                Some(EvalResult::G2(acc))
            }
        };

        values[out_idx] = value;
    }

    if values.iter().any(|v| v.is_none()) {
        tracing::error!(
            ast_nodes = ast.nodes.len(),
            hints = hints.len(),
            "Missing values while expanding deferred witnesses"
        );
        return Err(ProofVerifyError::default());
    }
    let values: Vec<EvalResult<BN254>> = values.into_iter().map(|v| v.unwrap()).collect();

    let get_g1 = |id: ValueId| -> &<BN254 as PairingCurve>::G1 {
        match &values[id.0 as usize] {
            EvalResult::G1(v) => v,
            _ => unreachable!("type mismatch for ValueId {id} (expected G1)"),
        }
    };
    let get_g2 = |id: ValueId| -> &<BN254 as PairingCurve>::G2 {
        match &values[id.0 as usize] {
            EvalResult::G2(v) => v,
            _ => unreachable!("type mismatch for ValueId {id} (expected G2)"),
        }
    };
    let get_gt = |id: ValueId| -> &<BN254 as PairingCurve>::GT {
        match &values[id.0 as usize] {
            EvalResult::GT(v) => v,
            _ => unreachable!("type mismatch for ValueId {id} (expected GT)"),
        }
    };

    use rayon::prelude::*;

    let g1_add: HashMap<OpId, <JoltWitness as WitnessBackend>::G1AddWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::G1Add {
                op_id: Some(op_id),
                a,
                b,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::G1(v) => v,
                    _ => unreachable!("G1Add output must be G1"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_g1_add(get_g1(*a), get_g1(*b), result),
                ))
            }
            _ => None,
        })
        .collect();

    let g1_scalar_mul: HashMap<OpId, <JoltWitness as WitnessBackend>::G1ScalarMulWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::G1ScalarMul {
                op_id: Some(op_id),
                point,
                scalar,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::G1(v) => v,
                    _ => unreachable!("G1ScalarMul output must be G1"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_g1_scalar_mul(
                        get_g1(*point),
                        &scalar.value,
                        result,
                    ),
                ))
            }
            _ => None,
        })
        .collect();

    let msm_g1: HashMap<OpId, <JoltWitness as WitnessBackend>::MsmG1Witness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::MsmG1 {
                op_id: Some(op_id),
                points,
                scalars,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::G1(v) => v,
                    _ => unreachable!("MsmG1 output must be G1"),
                };
                let bases: Vec<<BN254 as PairingCurve>::G1> =
                    points.iter().map(|p| *get_g1(*p)).collect();
                let scalars: Vec<<<BN254 as PairingCurve>::G1 as Group>::Scalar> =
                    scalars.iter().map(|s| s.value).collect();
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_msm_g1(&bases, &scalars, result),
                ))
            }
            _ => None,
        })
        .collect();

    let g2_add: HashMap<OpId, <JoltWitness as WitnessBackend>::G2AddWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::G2Add {
                op_id: Some(op_id),
                a,
                b,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::G2(v) => v,
                    _ => unreachable!("G2Add output must be G2"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_g2_add(get_g2(*a), get_g2(*b), result),
                ))
            }
            _ => None,
        })
        .collect();

    let g2_scalar_mul: HashMap<OpId, <JoltWitness as WitnessBackend>::G2ScalarMulWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::G2ScalarMul {
                op_id: Some(op_id),
                point,
                scalar,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::G2(v) => v,
                    _ => unreachable!("G2ScalarMul output must be G2"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_g2_scalar_mul(
                        get_g2(*point),
                        &scalar.value,
                        result,
                    ),
                ))
            }
            _ => None,
        })
        .collect();

    let msm_g2: HashMap<OpId, <JoltWitness as WitnessBackend>::MsmG2Witness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::MsmG2 {
                op_id: Some(op_id),
                points,
                scalars,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::G2(v) => v,
                    _ => unreachable!("MsmG2 output must be G2"),
                };
                let bases: Vec<<BN254 as PairingCurve>::G2> =
                    points.iter().map(|p| *get_g2(*p)).collect();
                let scalars: Vec<<<BN254 as PairingCurve>::G1 as Group>::Scalar> =
                    scalars.iter().map(|s| s.value).collect();
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_msm_g2(&bases, &scalars, result),
                ))
            }
            _ => None,
        })
        .collect();

    let gt_mul: HashMap<OpId, <JoltWitness as WitnessBackend>::GtMulWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::GTMul {
                op_id: Some(op_id),
                lhs,
                rhs,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::GT(v) => v,
                    _ => unreachable!("GTMul output must be GT"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_gt_mul(get_gt(*lhs), get_gt(*rhs), result),
                ))
            }
            _ => None,
        })
        .collect();

    let gt_exp: HashMap<OpId, <JoltWitness as WitnessBackend>::GtExpWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::GTExp {
                op_id: Some(op_id),
                base,
                scalar,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::GT(v) => v,
                    _ => unreachable!("GTExp output must be GT"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_gt_exp(get_gt(*base), &scalar.value, result),
                ))
            }
            _ => None,
        })
        .collect();

    let pairing: HashMap<OpId, <JoltWitness as WitnessBackend>::PairingWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::Pairing {
                op_id: Some(op_id),
                g1,
                g2,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::GT(v) => v,
                    _ => unreachable!("Pairing output must be GT"),
                };
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_pairing(get_g1(*g1), get_g2(*g2), result),
                ))
            }
            _ => None,
        })
        .collect();

    let multi_pairing: HashMap<OpId, <JoltWitness as WitnessBackend>::MultiPairingWitness> = ast
        .nodes
        .par_iter()
        .filter_map(|node| match &node.op {
            AstOp::MultiPairing {
                op_id: Some(op_id),
                g1s,
                g2s,
            } => {
                let result = match &values[node.out.0 as usize] {
                    EvalResult::GT(v) => v,
                    _ => unreachable!("MultiPairing output must be GT"),
                };
                let g1s: Vec<<BN254 as PairingCurve>::G1> =
                    g1s.iter().map(|id| *get_g1(*id)).collect();
                let g2s: Vec<<BN254 as PairingCurve>::G2> =
                    g2s.iter().map(|id| *get_g2(*id)).collect();
                Some((
                    *op_id,
                    JoltWitnessGenerator::generate_multi_pairing(&g1s, &g2s, result),
                ))
            }
            _ => None,
        })
        .collect();

    let mut witnesses = dory::recursion::WitnessCollection::<JoltWitness>::new();
    witnesses.num_rounds = hints.num_rounds;
    witnesses.g1_add = g1_add;
    witnesses.g1_scalar_mul = g1_scalar_mul;
    witnesses.msm_g1 = msm_g1;
    witnesses.g2_add = g2_add;
    witnesses.g2_scalar_mul = g2_scalar_mul;
    witnesses.msm_g2 = msm_g2;
    witnesses.gt_mul = gt_mul;
    witnesses.gt_exp = gt_exp;
    witnesses.pairing = pairing;
    witnesses.multi_pairing = multi_pairing;
    Ok(witnesses)
}

impl RecursionExt<Fr> for DoryCommitmentScheme {
    type Witness = WitnessCollection<JoltWitness>;
    type Ast = AstGraph<BN254>;
    type CombineHint = ArkGT;

    fn replay_opening_proof_transcript<ProofTranscript: crate::transcripts::Transcript>(
        proof: &ArkDoryProof,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Phase 0 only: derive Fiat–Shamir challenges on the real transcript (hashing).
        // This mutates `transcript` to the exact same final state as a normal verification run.
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        precompute_challenges::<ArkFr, BN254, _>(proof, &mut dory_transcript)
            .map_err(|_| ProofVerifyError::default())?;
        Ok(())
    }

    fn witness_gen_with_ast<ProofTranscript: crate::transcripts::Transcript>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<Fr as crate::field::JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<(Self::Witness, Self::Ast), ProofVerifyError> {
        // Convert Jolt types to dory types
        let reordered_point = reorder_opening_point_for_layout::<Fr>(point);
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev() // Reverse for dory endianness
            .map(|c| {
                let f_val: Fr = (*c).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_evaluation = jolt_to_ark(evaluation);

        let dory_setup: dory::setup::VerifierSetup<BN254> = setup.clone().into();

        // Run Dory verification with tracing enabled.
        //
        // IMPORTANT: `verify_recursive` is fundamentally sequential, but constructing Jolt's
        // recursion witnesses (packed traces, quotient polynomials, etc.) is very expensive.
        // We therefore record only minimal per-op inputs/outputs during verification and
        // expand them into full witnesses in parallel afterwards.
        let ctx = Rc::new(TraceContext::<
            DeferredJoltWitness,
            BN254,
            DeferredJoltWitnessGenerator,
        >::for_witness_gen_with_ast());
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
            *commitment,
            ark_evaluation,
            &ark_point,
            proof,
            dory_setup,
            &mut dory_transcript,
            ctx.clone(),
        )
        .map_err(|_| ProofVerifyError::default())?;

        let (witnesses_opt, ast_opt) = Rc::try_unwrap(ctx)
            .ok()
            .expect("TraceContext must not be shared after verify_recursive")
            .finalize_with_ast();

        let deferred_witnesses = witnesses_opt.ok_or(ProofVerifyError::default())?;
        let ast = ast_opt.ok_or(ProofVerifyError::default())?;

        let witnesses = tracing::info_span!("expand_deferred_dory_witnesses").in_scope(|| {
            Ok::<_, ProofVerifyError>(expand_deferred_witnesses(deferred_witnesses))
        })?;

        Ok((witnesses, ast))
    }

    fn build_symbolic_ast<ProofTranscript: crate::transcripts::Transcript>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<Fr as crate::field::JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<Self::Ast, ProofVerifyError> {
        // Convert point for dory
        let reordered_point = reorder_opening_point_for_layout::<Fr>(point);
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev() // Reverse for dory endianness
            .map(|c| {
                let f_val: Fr = (*c).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_evaluation = jolt_to_ark(evaluation);

        // Symbolic mode: no expensive group ops, but the AST records the full verification DAG.
        let ctx = Rc::new(TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_symbolic());
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
            *commitment,
            ark_evaluation,
            &ark_point,
            proof,
            setup.clone().into(),
            &mut dory_transcript,
            ctx.clone(),
        )
        .map_err(|_| ProofVerifyError::default())?;
        ctx.take_ast().ok_or(ProofVerifyError::default())
    }

    fn generate_combine_witness<C: std::borrow::Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Fr],
    ) -> (GTCombineWitness, Self::CombineHint) {
        assert!(!commitments.is_empty(), "commitments cannot be empty");
        assert_eq!(
            commitments.len(),
            coeffs.len(),
            "commitments and coeffs must have same length"
        );

        // Step 1: Generate exponentiation witnesses for each coeff * commitment.
        //
        // Note: `commitments` is generic and may not be `Sync`, so first copy out the small
        // `Fq12` values to enable parallel iteration on host without changing trait bounds.
        let comms_fq12: Vec<Fq12> = commitments.iter().map(|c| c.borrow().0).collect();

        let exp_witnesses: Vec<GTExpOpWitness> = {
            #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
            {
                comms_fq12
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(comm_fq12, coeff)| {
                        let exp_steps = Base4ExponentiationSteps::new(*comm_fq12, *coeff);
                        GTExpOpWitness {
                            base: exp_steps.base,
                            exponent: exp_steps.exponent,
                            result: exp_steps.result,
                            rho_mles: exp_steps.rho_mles,
                            quotient_mles: exp_steps.quotient_mles,
                            bits: exp_steps.bits,
                        }
                    })
                    .collect()
            }
            #[cfg(not(any(target_arch = "riscv64", target_arch = "riscv32")))]
            {
                use rayon::prelude::*;
                comms_fq12
                    .par_iter()
                    .zip(coeffs.par_iter())
                    .map(|(comm_fq12, coeff)| {
                        let exp_steps = Base4ExponentiationSteps::new(*comm_fq12, *coeff);
                        GTExpOpWitness {
                            base: exp_steps.base,
                            exponent: exp_steps.exponent,
                            result: exp_steps.result,
                            rho_mles: exp_steps.rho_mles,
                            quotient_mles: exp_steps.quotient_mles,
                            bits: exp_steps.bits,
                        }
                    })
                    .collect()
            }
        };

        // Step 2: Balanced binary-tree fold (associative group op) to expose parallelism.
        //
        // We record one `GTMulOpWitness` for each internal node, grouped by level.
        // The tree shape is deterministic given `exp_witnesses.len()`.
        let mut mul_layers: Vec<Vec<GTMulOpWitness>> = Vec::new();
        let mut layer: Vec<Fq12> = exp_witnesses.iter().map(|w| w.result).collect();

        while layer.len() > 1 {
            #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
            {
                let mut next = Vec::with_capacity((layer.len() + 1) / 2);
                let mut this_level_wits = Vec::with_capacity(layer.len() / 2);
                for chunk in layer.chunks(2) {
                    if let [a, b] = chunk {
                        let mul_steps = MultiplicationSteps::new(*a, *b);
                        this_level_wits.push(GTMulOpWitness {
                            lhs: mul_steps.lhs,
                            rhs: mul_steps.rhs,
                            result: mul_steps.result,
                            quotient_mle: mul_steps.quotient_mle,
                        });
                        next.push(mul_steps.result);
                    } else {
                        // Odd tail element: carry forward.
                        next.push(chunk[0]);
                    }
                }
                mul_layers.push(this_level_wits);
                layer = next;
            }

            #[cfg(not(any(target_arch = "riscv64", target_arch = "riscv32")))]
            {
                use rayon::prelude::*;

                // IMPORTANT: `par_chunks(2)` is an IndexedParallelIterator, so collecting into a Vec
                // preserves left-to-right order deterministically.
                let pairs: Vec<(Fq12, Option<GTMulOpWitness>)> = layer
                    .par_chunks(2)
                    .map(|chunk| {
                        if let [a, b] = chunk {
                            let mul_steps = MultiplicationSteps::new(*a, *b);
                            (
                                mul_steps.result,
                                Some(GTMulOpWitness {
                                    lhs: mul_steps.lhs,
                                    rhs: mul_steps.rhs,
                                    result: mul_steps.result,
                                    quotient_mle: mul_steps.quotient_mle,
                                }),
                            )
                        } else {
                            // Odd tail element: carry forward.
                            (chunk[0], None)
                        }
                    })
                    .collect();

                let mut next = Vec::with_capacity(pairs.len());
                let mut this_level_wits = Vec::with_capacity(layer.len() / 2);
                for (res, wit) in pairs {
                    next.push(res);
                    if let Some(w) = wit {
                        this_level_wits.push(w);
                    }
                }
                mul_layers.push(this_level_wits);
                layer = next;
            }
        }

        let accumulator = layer[0];

        let witness = GTCombineWitness {
            exp_witnesses,
            mul_layers,
        };
        debug_assert_eq!(
            CombineDag::new(witness.exp_witnesses.len())
                .layers
                .iter()
                .map(|l| l.muls.len())
                .collect::<Vec<_>>(),
            witness
                .mul_layers
                .iter()
                .map(|l| l.len())
                .collect::<Vec<_>>(),
            "GTCombineWitness.mul_layers shape must match CombineDag"
        );
        debug_assert!(
            witness.validate_tree_wiring().is_ok(),
            "GTCombineWitness tree wiring invalid"
        );

        // The hint is the final accumulated result
        let hint = ArkGT(accumulator);

        (witness, hint)
    }

    fn combine_with_hint(hint: &Self::CombineHint) -> Self::Commitment {
        *hint
    }

    fn combine_hint_to_fq12(hint: &Self::CombineHint) -> Fq12 {
        hint.0
    }

    fn combine_with_hint_fq12(hint: &Fq12) -> Self::Commitment {
        ArkGT(*hint)
    }

    fn derive_pairing_boundary_from_ast(
        ast: &Self::Ast,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        joint_commitment: Self::Commitment,
        combine_commitments: &[Self::Commitment],
        combine_coeffs: &[Fr],
    ) -> Result<PairingBoundary, ProofVerifyError> {
        let derived = crate::poly::commitment::dory::derive_from_dory_ast(
            ast,
            proof,
            setup,
            joint_commitment,
            combine_commitments,
            combine_coeffs,
        )?;
        Ok(derived.pairing_boundary)
    }
}

// ============================================================================
// AST-enabled witness generation
// ============================================================================

/// Result of witness generation with AST tracing enabled.
///
/// Contains everything needed for AST-ordered constraint building and wiring.
#[cfg(any())]
pub struct WitnessWithAst {
    /// Collected witnesses from all operations
    pub witnesses: dory::recursion::WitnessCollection<JoltWitness>,
    /// AST graph capturing the computation DAG
    pub ast: AstGraph<BN254>,
    /// Hints for efficient verification
    pub hints: JoltHintMap,
}

/// Generate witnesses with AST tracing enabled.
///
/// This is the entry point for recursion provers that need the AST for wiring constraints.
/// The AST captures the full computation DAG, enabling:
/// - Deterministic constraint ordering (by topological sort of AST nodes)
/// - Wiring constraint derivation (from AST edges)
/// - Boundary constraint identification (from AST roots and leaves)
#[cfg(any())]
pub fn witness_gen_with_ast<ProofTranscript: crate::transcripts::Transcript>(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut ProofTranscript,
    point: &[<Fr as crate::field::JoltField>::Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
) -> Result<WitnessWithAst, ProofVerifyError> {
    // Convert Jolt types to dory types
    let reordered_point =
        crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout::<Fr>(
            point,
        );
    let ark_point: Vec<ArkFr> = reordered_point
        .iter()
        .rev() // Reverse for dory endianness
        .map(|c| {
            let f_val: Fr = (*c).into();
            jolt_to_ark(&f_val)
        })
        .collect();
    let ark_evaluation = jolt_to_ark(evaluation);

    let dory_setup: dory::setup::VerifierSetup<BN254> = setup.clone().into();

    // Phase 0: derive Fiat–Shamir challenges on the real transcript (hashing).
    // This mutates `transcript` to the exact same final state as a normal verification run.
    let mut dory_transcript = JoltToDoryTranscript::new(transcript);
    let challenges = precompute_challenges::<ArkFr, BN254, _>(proof, &mut dory_transcript)
        .map_err(|_| ProofVerifyError::default())?;

    // Phase 1: build verification AST with challenge replay (no transcript hashing, no group ops).
    let mut backend = AstOnlyBackend::new();
    let mut replay_transcript = ChallengeReplayTranscript::new(challenges);
    verify_with_backend::<ArkFr, BN254, _, _>(
        *commitment,
        ark_evaluation,
        &ark_point,
        proof,
        dory_setup.clone(),
        &mut replay_transcript,
        &mut backend,
    )
    .map_err(|_| ProofVerifyError::default())?;

    let ast = backend.finalize();
    debug_assert!(ast.validate().is_ok(), "Dory verification AST invalid");

    // Phase 2: evaluate AST in parallel to recover all op outputs (hints).
    let input_provider = DoryInputProviderWithCommitment::new(&dory_setup, proof, *commitment);
    let ops = ArkworksOpEvaluator;
    let results = TaskExecutor::new(&ast, &input_provider, &ops).execute();

    let mut hints = HintMap::new(proof.sigma);
    for (op_id, value_id) in ast.opid_to_value.iter() {
        let v = results
            .get(value_id)
            .unwrap_or_else(|| panic!("Missing AST eval result for {value_id}"));
        match v {
            EvalResult::G1(g1) => hints.insert_g1(*op_id, *g1),
            EvalResult::G2(g2) => hints.insert_g2(*op_id, *g2),
            EvalResult::GT(gt) => hints.insert_gt(*op_id, *gt),
        }
    }

    // Phase 3: expand detailed witnesses in parallel from (AST + outputs-only hints).
    let witnesses = expand_witnesses_from_deferred(&ast, &hints, &dory_setup, proof, *commitment)?;
    let hints = JoltHintMap(hints);

    Ok(WitnessWithAst {
        witnesses,
        ast,
        hints,
    })
}

/// Reconstruct AST from public inputs (verifier-side).
///
/// This re-runs verification with AST tracing but uses hints instead of
/// computing the expensive operations. The resulting AST is deterministic
/// and matches the prover's AST.
#[cfg(any())]
pub fn reconstruct_ast<ProofTranscript: crate::transcripts::Transcript>(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut ProofTranscript,
    point: &[<Fr as crate::field::JoltField>::Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
    hints: &JoltHintMap,
) -> Result<AstGraph<BN254>, ProofVerifyError> {
    // Convert Jolt types to dory types
    let reordered_point =
        crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout::<Fr>(
            point,
        );
    let ark_point: Vec<ArkFr> = reordered_point
        .iter()
        .rev()
        .map(|c| {
            let f_val: Fr = (*c).into();
            jolt_to_ark(&f_val)
        })
        .collect();
    let ark_evaluation = jolt_to_ark(evaluation);

    // Create hint-based context WITH AST tracing
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_hints(hints.0.clone())
            .with_ast(),
    );

    // Wrap transcript for dory compatibility
    let mut dory_transcript = JoltToDoryTranscript::new(transcript);

    // Run verification with hints (fast) while building AST
    verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
        *commitment,
        ark_evaluation,
        &ark_point,
        proof,
        setup.clone().into(),
        &mut dory_transcript,
        ctx.clone(),
    )
    .map_err(|_| ProofVerifyError::default())?;

    // Extract only the AST (witnesses are None in hint mode)
    let (_, ast_opt) = Rc::try_unwrap(ctx)
        .ok()
        .expect("Should have sole ownership")
        .finalize_with_ast();

    ast_opt.ok_or(ProofVerifyError::default())
}
