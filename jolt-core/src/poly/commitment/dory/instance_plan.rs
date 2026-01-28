//! Derive recursion verifier inputs from Dory's symbolic AST.
//!
//! This is the "no-hints" path: the verifier deterministically reconstructs the recursion
//! instance plan (constraint ordering + public inputs) from the Dory verification AST, rather
//! than trusting prover-supplied metadata.

use super::wrappers::{
    ark_to_jolt, ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksVerifierSetup, BN254,
};
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::proof_serialization::NonInputBaseHints;
use crate::zkvm::proof_serialization::PairingBoundary;
use crate::zkvm::recursion::constraints::system::ConstraintType;
use crate::zkvm::recursion::g1::scalar_multiplication::G1ScalarMulPublicInputs;
use crate::zkvm::recursion::g2::scalar_multiplication::G2ScalarMulPublicInputs;
use crate::zkvm::recursion::gt::exponentiation::GtExpPublicInputs;
use crate::zkvm::recursion::prefix_packing::PrefixPackingLayout;
use crate::zkvm::recursion::verifier::RecursionVerifierInput;
use crate::zkvm::recursion::wiring_plan::derive_wiring_plan;
use crate::zkvm::recursion::CombineDag;
use crate::zkvm::recursion::PolyType;

use ark_bn254::{Fq12, Fr, G1Affine, G2Affine};
use ark_ec::CurveGroup;
use ark_ff::{PrimeField, Zero};
use dory::primitives::arithmetic::{Group as DoryGroup, PairingCurve};
use dory::recursion::ast::{
    AstConstraint, AstGraph, AstOp, InputSource, RoundMsg, ValueId, ValueType,
};
use dory::recursion::OpId;
use dory::setup::VerifierSetup;

#[derive(Clone, Debug)]
enum Value {
    G1(ArkG1),
    G2(ArkG2),
    GT(ArkGT),
}

/// Convert a scalar exponent to MSB-first bits with no leading zeros.
///
/// Matches `Base4ExponentiationSteps` in `poly/commitment/dory/witness/gt_exp.rs`.
pub fn bits_from_exponent_msb_no_leading_zeros(exponent: Fr) -> Vec<bool> {
    let mut n = exponent.into_bigint();
    if n.is_zero() {
        return vec![];
    }

    // Base-4 digits, lsb-first, then reverse to msb-first.
    let mut digits_lsb = Vec::new();
    while !n.is_zero() {
        let limb0 = n.as_ref()[0];
        digits_lsb.push((limb0 & 3) as u8);
        n >>= 2;
    }
    digits_lsb.reverse();

    // Convert base-4 digits to MSB-first bits with no leading zeros.
    let mut bits = Vec::with_capacity(digits_lsb.len() * 2);
    let mut started = false;
    for &digit in &digits_lsb {
        let hi = (digit & 2) != 0;
        let lo = (digit & 1) != 0;
        if !started {
            if hi {
                bits.push(true);
                bits.push(lo);
                started = true;
            } else if lo {
                bits.push(true);
                started = true;
            }
        } else {
            bits.push(hi);
            bits.push(lo);
        }
    }
    bits
}

fn resolve_input_g1(
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    src: &InputSource,
) -> Result<ArkG1, ProofVerifyError> {
    match src {
        InputSource::Setup { name, index } => match (*name, index) {
            ("g1_0", None) => Ok(setup.g1_0),
            ("h1", None) => Ok(setup.h1),
            _ => Err(ProofVerifyError::default()),
        },
        InputSource::Proof { name } => match *name {
            "vmv.e1" | "vmv.e1_init" => Ok(proof.vmv_message.e1),
            "final.e1" => Ok(proof.final_message.e1),
            _ => Err(ProofVerifyError::default()),
        },
        InputSource::ProofRound { round, msg, name } => {
            // Bounds check (zero runtime cost in release builds)
            debug_assert!(
                *round < proof.first_messages.len(),
                "round {} out of bounds (first_messages len {})",
                round,
                proof.first_messages.len()
            );
            debug_assert!(
                *round < proof.second_messages.len(),
                "round {} out of bounds (second_messages len {})",
                round,
                proof.second_messages.len()
            );
            match msg {
                RoundMsg::First => match *name {
                    "e1_beta" => Ok(proof.first_messages[*round].e1_beta),
                    _ => Err(ProofVerifyError::default()),
                },
                RoundMsg::Second => match *name {
                    "e1_plus" => Ok(proof.second_messages[*round].e1_plus),
                    "e1_minus" => Ok(proof.second_messages[*round].e1_minus),
                    _ => Err(ProofVerifyError::default()),
                },
            }
        }
    }
}

fn resolve_input_g2(
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    src: &InputSource,
) -> Result<ArkG2, ProofVerifyError> {
    match src {
        InputSource::Setup { name, index } => match (*name, index) {
            ("g2_0", None) => Ok(setup.g2_0),
            ("h2", None) => Ok(setup.h2),
            _ => Err(ProofVerifyError::default()),
        },
        InputSource::Proof { name } => match *name {
            "final.e2" => Ok(proof.final_message.e2),
            _ => Err(ProofVerifyError::default()),
        },
        InputSource::ProofRound { round, msg, name } => {
            debug_assert!(
                *round < proof.first_messages.len(),
                "round {} out of bounds (first_messages len {})",
                round,
                proof.first_messages.len()
            );
            debug_assert!(
                *round < proof.second_messages.len(),
                "round {} out of bounds (second_messages len {})",
                round,
                proof.second_messages.len()
            );
            match msg {
                RoundMsg::First => match *name {
                    "e2_beta" => Ok(proof.first_messages[*round].e2_beta),
                    _ => Err(ProofVerifyError::default()),
                },
                RoundMsg::Second => match *name {
                    "e2_plus" => Ok(proof.second_messages[*round].e2_plus),
                    "e2_minus" => Ok(proof.second_messages[*round].e2_minus),
                    _ => Err(ProofVerifyError::default()),
                },
            }
        }
    }
}

fn resolve_input_gt(
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    commitment: ArkGT,
    src: &InputSource,
) -> Result<ArkGT, ProofVerifyError> {
    match src {
        InputSource::Setup { name, index } => match (*name, index) {
            ("chi", Some(i)) => {
                debug_assert!(*i < setup.chi.len(), "chi index {} out of bounds", i);
                Ok(setup.chi[*i])
            }
            ("delta_1l", Some(i)) => {
                debug_assert!(
                    *i < setup.delta_1l.len(),
                    "delta_1l index {} out of bounds",
                    i
                );
                Ok(setup.delta_1l[*i])
            }
            ("delta_1r", Some(i)) => {
                debug_assert!(
                    *i < setup.delta_1r.len(),
                    "delta_1r index {} out of bounds",
                    i
                );
                Ok(setup.delta_1r[*i])
            }
            ("delta_2l", Some(i)) => {
                debug_assert!(
                    *i < setup.delta_2l.len(),
                    "delta_2l index {} out of bounds",
                    i
                );
                Ok(setup.delta_2l[*i])
            }
            ("delta_2r", Some(i)) => {
                debug_assert!(
                    *i < setup.delta_2r.len(),
                    "delta_2r index {} out of bounds",
                    i
                );
                Ok(setup.delta_2r[*i])
            }
            ("ht", None) => Ok(setup.ht),
            _ => Err(ProofVerifyError::default()),
        },
        InputSource::Proof { name } => match *name {
            "vmv.c" => Ok(proof.vmv_message.c),
            "vmv.d2" | "vmv.d2_init" => Ok(proof.vmv_message.d2),
            "commitment" => Ok(commitment),
            _ => Err(ProofVerifyError::default()),
        },
        InputSource::ProofRound { round, msg, name } => {
            debug_assert!(
                *round < proof.first_messages.len(),
                "round {} out of bounds (first_messages len {})",
                round,
                proof.first_messages.len()
            );
            debug_assert!(
                *round < proof.second_messages.len(),
                "round {} out of bounds (second_messages len {})",
                round,
                proof.second_messages.len()
            );
            match msg {
                RoundMsg::First => match *name {
                    "d1_left" => Ok(proof.first_messages[*round].d1_left),
                    "d1_right" => Ok(proof.first_messages[*round].d1_right),
                    "d2_left" => Ok(proof.first_messages[*round].d2_left),
                    "d2_right" => Ok(proof.first_messages[*round].d2_right),
                    _ => Err(ProofVerifyError::default()),
                },
                RoundMsg::Second => match *name {
                    "c_plus" => Ok(proof.second_messages[*round].c_plus),
                    "c_minus" => Ok(proof.second_messages[*round].c_minus),
                    _ => Err(ProofVerifyError::default()),
                },
            }
        }
    }
}

fn eval_ast(
    ast: &AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    commitment: ArkGT,
) -> Result<Vec<Option<Value>>, ProofVerifyError> {
    let mut values: Vec<Option<Value>> = vec![None; ast.nodes.len()];

    for node in &ast.nodes {
        let out_idx = node.out.0 as usize;
        debug_assert!(
            out_idx < values.len(),
            "AST node output index {} out of bounds (len {})",
            out_idx,
            values.len()
        );

        // Helper to safely access values with debug assertion
        let get_value = |id: ValueId| -> Result<&Value, ProofVerifyError> {
            let idx = id.0 as usize;
            debug_assert!(
                idx < values.len(),
                "ValueId {} out of bounds (len {})",
                idx,
                values.len()
            );
            values[idx].as_ref().ok_or(ProofVerifyError::default())
        };

        let v = match (&node.out_ty, &node.op) {
            (ValueType::G1, AstOp::Input { source }) => {
                Value::G1(resolve_input_g1(proof, setup, source)?)
            }
            (ValueType::G2, AstOp::Input { source }) => {
                Value::G2(resolve_input_g2(proof, setup, source)?)
            }
            (ValueType::GT, AstOp::Input { source }) => {
                Value::GT(resolve_input_gt(proof, setup, commitment, source)?)
            }

            (ValueType::G1, AstOp::G1Add { a, b, .. }) => {
                let a_val = match get_value(*a)? {
                    Value::G1(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                let b_val = match get_value(*b)? {
                    Value::G1(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::G1(a_val.add(b_val))
            }
            (ValueType::G1, AstOp::G1ScalarMul { point, scalar, .. }) => {
                let p = match get_value(*point)? {
                    Value::G1(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::G1(p.scale(&scalar.value))
            }

            (ValueType::G2, AstOp::G2Add { a, b, .. }) => {
                let a_val = match get_value(*a)? {
                    Value::G2(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                let b_val = match get_value(*b)? {
                    Value::G2(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::G2(a_val.add(b_val))
            }
            (ValueType::G2, AstOp::G2ScalarMul { point, scalar, .. }) => {
                let p = match get_value(*point)? {
                    Value::G2(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::G2(p.scale(&scalar.value))
            }

            (ValueType::GT, AstOp::GTMul { lhs, rhs, .. }) => {
                let a = match get_value(*lhs)? {
                    Value::GT(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                let b = match get_value(*rhs)? {
                    Value::GT(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::GT(a.add(b))
            }
            (ValueType::GT, AstOp::GTExp { base, scalar, .. }) => {
                let b = match get_value(*base)? {
                    Value::GT(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::GT(b.scale(&scalar.value))
            }

            // Pairing nodes are only needed for the final equality check; we still evaluate them
            // so downstream consumers can choose whether to recompute or use extracted boundary.
            (ValueType::GT, AstOp::Pairing { g1, g2, .. }) => {
                let p = match get_value(*g1)? {
                    Value::G1(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                let q = match get_value(*g2)? {
                    Value::G2(v) => v,
                    _ => return Err(ProofVerifyError::default()),
                };
                Value::GT(BN254::pair(p, q))
            }
            (ValueType::GT, AstOp::MultiPairing { g1s, g2s, .. }) => {
                let mut ps = Vec::with_capacity(g1s.len());
                let mut qs = Vec::with_capacity(g2s.len());
                for id in g1s {
                    let p = match get_value(*id)? {
                        Value::G1(v) => v,
                        _ => return Err(ProofVerifyError::default()),
                    };
                    ps.push(*p);
                }
                for id in g2s {
                    let q = match get_value(*id)? {
                        Value::G2(v) => v,
                        _ => return Err(ProofVerifyError::default()),
                    };
                    qs.push(*q);
                }
                Value::GT(BN254::multi_pair(&ps, &qs))
            }

            // MSMs are not currently supported by Jolt recursion constraints.
            (_, AstOp::MsmG1 { .. } | AstOp::MsmG2 { .. }) => {
                return Err(ProofVerifyError::default())
            }

            _ => return Err(ProofVerifyError::default()),
        };

        values[out_idx] = Some(v);
    }

    Ok(values)
}

/// Result of deriving recursion input from a Dory AST.
pub struct DerivedRecursionInput {
    /// The verifier input (constraint types + public inputs).
    pub verifier_input: RecursionVerifierInput,
    /// Number of variables for the dense polynomial (Hyrax sizing).
    pub dense_num_vars: usize,
    /// Pairing boundary for external 3-way multi-pairing check.
    pub pairing_boundary: PairingBoundary,
}

/// Result of deriving only the recursion verifier input (instance plan) from a Dory AST.
///
/// This intentionally excludes `PairingBoundary` extraction so callers can treat pairing inputs
/// as prover-supplied hints for performance (until wiring/boundary constraints exist).
pub struct DerivedRecursionPlan {
    pub verifier_input: RecursionVerifierInput,
    pub dense_num_vars: usize,
}

fn resolve_gt_input_or_hint(
    ast: &AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    joint_commitment: ArkGT,
    value_id: ValueId,
    hint: &Option<Fq12>,
) -> Result<Fq12, ProofVerifyError> {
    let idx = value_id.0 as usize;
    debug_assert!(idx < ast.nodes.len(), "ValueId out of bounds");
    match &ast.nodes[idx].op {
        AstOp::Input { source } => Ok(resolve_input_gt(proof, setup, joint_commitment, source)?.0),
        _ => hint.ok_or(ProofVerifyError::default()),
    }
}

fn resolve_g1_input_or_hint(
    ast: &AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    value_id: ValueId,
    hint: &Option<G1Affine>,
) -> Result<G1Affine, ProofVerifyError> {
    let idx = value_id.0 as usize;
    debug_assert!(idx < ast.nodes.len(), "ValueId out of bounds");
    match &ast.nodes[idx].op {
        AstOp::Input { source } => Ok(resolve_input_g1(proof, setup, source)?.0.into_affine()),
        _ => hint.ok_or(ProofVerifyError::default()),
    }
}

fn resolve_g2_input_or_hint(
    ast: &AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &VerifierSetup<BN254>,
    value_id: ValueId,
    hint: &Option<G2Affine>,
) -> Result<G2Affine, ProofVerifyError> {
    let idx = value_id.0 as usize;
    debug_assert!(idx < ast.nodes.len(), "ValueId out of bounds");
    match &ast.nodes[idx].op {
        AstOp::Input { source } => Ok(resolve_input_g2(proof, setup, source)?.0.into_affine()),
        _ => hint.ok_or(ProofVerifyError::default()),
    }
}

/// Derive the recursion verifier input from a Dory AST, using hints when a base/point is not an input.
pub fn derive_plan_with_hints(
    ast: &AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    joint_commitment: ArkGT,
    combine_commitments: &[ArkGT],
    combine_coeffs: &[Fr],
    non_input_hints: &NonInputBaseHints,
    pairing_boundary: PairingBoundary,
    joint_commitment_fq12: Fq12,
) -> Result<DerivedRecursionPlan, ProofVerifyError> {
    let dory_setup: VerifierSetup<BN254> = setup.clone().into();

    // Collect ops by type (sorted by OpId for Dory-traced operations).
    let mut gt_exp_ops: Vec<(OpId, ValueId, ArkFr)> = Vec::new();
    let mut gt_mul_ops: Vec<OpId> = Vec::new();
    let mut g1_scalar_mul_ops: Vec<(OpId, ValueId, ArkFr)> = Vec::new();
    let mut g2_scalar_mul_ops: Vec<(OpId, ValueId, ArkFr)> = Vec::new();
    let mut g1_add_ops: Vec<OpId> = Vec::new();
    let mut g2_add_ops: Vec<OpId> = Vec::new();

    for node in &ast.nodes {
        match &node.op {
            AstOp::GTExp {
                op_id: Some(id),
                base,
                scalar,
            } => gt_exp_ops.push((*id, *base, scalar.value)),
            AstOp::GTMul {
                op_id: Some(id), ..
            } => gt_mul_ops.push(*id),
            AstOp::G1ScalarMul {
                op_id: Some(id),
                point,
                scalar,
            } => g1_scalar_mul_ops.push((*id, *point, scalar.value)),
            AstOp::G2ScalarMul {
                op_id: Some(id),
                point,
                scalar,
            } => g2_scalar_mul_ops.push((*id, *point, scalar.value)),
            AstOp::G1Add {
                op_id: Some(id), ..
            } => g1_add_ops.push(*id),
            AstOp::G2Add {
                op_id: Some(id), ..
            } => g2_add_ops.push(*id),
            _ => {}
        }
    }

    gt_exp_ops.sort_by_key(|(id, _, _)| *id);
    gt_mul_ops.sort();
    g1_scalar_mul_ops.sort_by_key(|(id, _, _)| *id);
    g2_scalar_mul_ops.sort_by_key(|(id, _, _)| *id);
    g1_add_ops.sort();
    g2_add_ops.sort();

    if non_input_hints.gt_exp_base_hints.len() != gt_exp_ops.len()
        || non_input_hints.g1_scalar_mul_base_hints.len() != g1_scalar_mul_ops.len()
        || non_input_hints.g2_scalar_mul_base_hints.len() != g2_scalar_mul_ops.len()
    {
        return Err(ProofVerifyError::default());
    }

    let mut constraint_types: Vec<ConstraintType> = Vec::new();
    let mut gt_exp_public_inputs: Vec<GtExpPublicInputs> = Vec::new();
    let mut g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs> = Vec::new();
    let mut g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs> = Vec::new();

    // Dory GTExp
    for ((_, base_id, scalar), base_hint) in gt_exp_ops
        .iter()
        .zip(non_input_hints.gt_exp_base_hints.iter())
    {
        let base = resolve_gt_input_or_hint(
            ast,
            proof,
            &dory_setup,
            joint_commitment,
            *base_id,
            base_hint,
        )?;
        let exponent: Fr = ark_to_jolt(scalar);
        let bits = bits_from_exponent_msb_no_leading_zeros(exponent);
        constraint_types.push(ConstraintType::GtExp);
        gt_exp_public_inputs.push(GtExpPublicInputs::new(base, bits));
    }

    // Dory GTMul
    for _ in &gt_mul_ops {
        constraint_types.push(ConstraintType::GtMul);
    }

    // Dory G1 scalar mul
    for ((_, point_id, scalar), point_hint) in g1_scalar_mul_ops
        .iter()
        .zip(non_input_hints.g1_scalar_mul_base_hints.iter())
    {
        let p = resolve_g1_input_or_hint(ast, proof, &dory_setup, *point_id, point_hint)?;
        let scalar_fr: Fr = ark_to_jolt(scalar);
        constraint_types.push(ConstraintType::G1ScalarMul {
            base_point: (p.x, p.y),
        });
        g1_scalar_mul_public_inputs.push(G1ScalarMulPublicInputs::new(scalar_fr));
    }

    // Dory G2 scalar mul
    for ((_, point_id, scalar), point_hint) in g2_scalar_mul_ops
        .iter()
        .zip(non_input_hints.g2_scalar_mul_base_hints.iter())
    {
        let p = resolve_g2_input_or_hint(ast, proof, &dory_setup, *point_id, point_hint)?;
        let scalar_fr: Fr = ark_to_jolt(scalar);
        constraint_types.push(ConstraintType::G2ScalarMul {
            base_point: (p.x, p.y),
        });
        g2_scalar_mul_public_inputs.push(G2ScalarMulPublicInputs::new(scalar_fr));
    }

    // Dory adds
    for _ in &g1_add_ops {
        constraint_types.push(ConstraintType::G1Add);
    }
    for _ in &g2_add_ops {
        constraint_types.push(ConstraintType::G2Add);
    }

    // Combine commitments constraints are appended at the end (matching prover builder).
    for (commitment, coeff) in combine_commitments.iter().zip(combine_coeffs.iter()) {
        let bits = bits_from_exponent_msb_no_leading_zeros(*coeff);
        constraint_types.push(ConstraintType::GtExp);
        gt_exp_public_inputs.push(GtExpPublicInputs::new(commitment.0, bits));
    }
    let combine_mul_count = CombineDag::new(combine_commitments.len()).num_muls_total();
    for _ in 0..combine_mul_count {
        constraint_types.push(ConstraintType::GtMul);
    }

    let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false);
    let enable_g1_fused_wiring_end_to_end =
        std::env::var("JOLT_RECURSION_ENABLE_G1_FUSED_WIRING_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
    let enable_g1_scalar_mul_fused_end_to_end =
        std::env::var("JOLT_RECURSION_ENABLE_G1_SCALAR_MUL_FUSED_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
            || enable_g1_fused_wiring_end_to_end;
    let enable_g1_add_fused_end_to_end = enable_g1_fused_wiring_end_to_end;

    let dense_num_vars = if enable_gt_fused_end_to_end
        || enable_g1_scalar_mul_fused_end_to_end
        || enable_g1_add_fused_end_to_end
    {
        PrefixPackingLayout::from_constraint_types_fused(
            &constraint_types,
            enable_gt_fused_end_to_end,
            enable_g1_scalar_mul_fused_end_to_end,
            enable_g1_add_fused_end_to_end,
        )
        .num_dense_vars
    } else {
        PrefixPackingLayout::from_constraint_types(&constraint_types).num_dense_vars
    };

    let num_constraints = constraint_types.len();
    let num_constraints_padded = num_constraints.next_power_of_two();
    let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
    let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
    let num_constraint_vars = 11usize;
    let num_vars = num_s_vars + num_constraint_vars;

    let wiring = derive_wiring_plan(ast, combine_commitments.len(), &pairing_boundary)?;

    Ok(DerivedRecursionPlan {
        verifier_input: RecursionVerifierInput {
            constraint_types,
            enable_gt_fused_end_to_end,
            enable_g1_scalar_mul_fused_end_to_end,
            enable_g1_fused_wiring_end_to_end,
            num_vars,
            num_constraint_vars,
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            gt_exp_public_inputs,
            g1_scalar_mul_public_inputs,
            g2_scalar_mul_public_inputs,
            wiring,
            pairing_boundary,
            joint_commitment: joint_commitment_fq12,
        },
        dense_num_vars,
    })
}

/// Derive the recursion verifier input and pairing boundary from a Dory AST.
///
/// `combine_commitments` and `combine_coeffs` must be in the same deterministic order as the
/// recursion prover's `compute_rlc_coefficients` iteration (BTreeMap order).
#[tracing::instrument(skip_all)]
pub fn derive_from_dory_ast(
    ast: &AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    joint_commitment: ArkGT,
    combine_commitments: &[ArkGT],
    combine_coeffs: &[Fr],
) -> Result<DerivedRecursionInput, ProofVerifyError> {
    let dory_setup: VerifierSetup<BN254> = setup.clone().into();

    let values = eval_ast(ast, proof, &dory_setup, joint_commitment)?;

    // Collect ops by type (sorted by OpId for Dory-traced operations).
    let mut gt_exp_ops: Vec<(OpId, ValueId, ArkFr)> = Vec::new();
    let mut gt_mul_ops: Vec<OpId> = Vec::new();
    let mut g1_scalar_mul_ops: Vec<(OpId, ValueId, ArkFr)> = Vec::new();
    let mut g2_scalar_mul_ops: Vec<(OpId, ValueId, ArkFr)> = Vec::new();
    let mut g1_add_ops: Vec<OpId> = Vec::new();
    let mut g2_add_ops: Vec<OpId> = Vec::new();

    for node in &ast.nodes {
        match &node.op {
            AstOp::GTExp {
                op_id: Some(id),
                base,
                scalar,
            } => gt_exp_ops.push((*id, *base, scalar.value)),
            AstOp::GTMul {
                op_id: Some(id), ..
            } => gt_mul_ops.push(*id),
            AstOp::G1ScalarMul {
                op_id: Some(id),
                point,
                scalar,
            } => g1_scalar_mul_ops.push((*id, *point, scalar.value)),
            AstOp::G2ScalarMul {
                op_id: Some(id),
                point,
                scalar,
            } => g2_scalar_mul_ops.push((*id, *point, scalar.value)),
            AstOp::G1Add {
                op_id: Some(id), ..
            } => g1_add_ops.push(*id),
            AstOp::G2Add {
                op_id: Some(id), ..
            } => g2_add_ops.push(*id),
            _ => {}
        }
    }

    gt_exp_ops.sort_by_key(|(id, _, _)| *id);
    gt_mul_ops.sort();
    g1_scalar_mul_ops.sort_by_key(|(id, _, _)| *id);
    g2_scalar_mul_ops.sort_by_key(|(id, _, _)| *id);
    g1_add_ops.sort();
    g2_add_ops.sort();

    let mut constraint_types: Vec<ConstraintType> = Vec::new();
    let mut gt_exp_public_inputs: Vec<GtExpPublicInputs> = Vec::new();
    let mut g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs> = Vec::new();
    let mut g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs> = Vec::new();

    // Dory GTExp
    for (_op_id, base_id, scalar) in &gt_exp_ops {
        let idx = base_id.0 as usize;
        debug_assert!(idx < values.len(), "base_id {} out of bounds", idx);
        let base = match values[idx].as_ref().ok_or(ProofVerifyError::default())? {
            Value::GT(v) => v.0,
            _ => return Err(ProofVerifyError::default()),
        };
        let exponent: Fr = ark_to_jolt(scalar);
        let bits = bits_from_exponent_msb_no_leading_zeros(exponent);
        constraint_types.push(ConstraintType::GtExp);
        gt_exp_public_inputs.push(GtExpPublicInputs::new(base, bits));
    }

    // Dory GTMul
    for _ in &gt_mul_ops {
        constraint_types.push(ConstraintType::GtMul);
    }

    // Dory G1 scalar mul
    for (_op_id, point_id, scalar) in &g1_scalar_mul_ops {
        let idx = point_id.0 as usize;
        debug_assert!(idx < values.len(), "point_id {} out of bounds", idx);
        let p = match values[idx].as_ref().ok_or(ProofVerifyError::default())? {
            Value::G1(v) => v.0.into_affine(),
            _ => return Err(ProofVerifyError::default()),
        };
        let scalar_fr: Fr = ark_to_jolt(scalar);
        constraint_types.push(ConstraintType::G1ScalarMul {
            base_point: (p.x, p.y),
        });
        g1_scalar_mul_public_inputs.push(G1ScalarMulPublicInputs::new(scalar_fr));
    }

    // Dory G2 scalar mul
    for (_op_id, point_id, scalar) in &g2_scalar_mul_ops {
        let idx = point_id.0 as usize;
        debug_assert!(idx < values.len(), "point_id {} out of bounds", idx);
        let p = match values[idx].as_ref().ok_or(ProofVerifyError::default())? {
            Value::G2(v) => v.0.into_affine(),
            _ => return Err(ProofVerifyError::default()),
        };
        let scalar_fr: Fr = ark_to_jolt(scalar);
        constraint_types.push(ConstraintType::G2ScalarMul {
            base_point: (p.x, p.y),
        });
        g2_scalar_mul_public_inputs.push(G2ScalarMulPublicInputs::new(scalar_fr));
    }

    // Dory adds
    for _ in &g1_add_ops {
        constraint_types.push(ConstraintType::G1Add);
    }
    for _ in &g2_add_ops {
        constraint_types.push(ConstraintType::G2Add);
    }

    // Combine commitments constraints are appended at the end (matching prover builder).
    // First: one GTExp per combined term.
    for (commitment, coeff) in combine_commitments.iter().zip(combine_coeffs.iter()) {
        let bits = bits_from_exponent_msb_no_leading_zeros(*coeff);
        constraint_types.push(ConstraintType::GtExp);
        gt_exp_public_inputs.push(GtExpPublicInputs::new(commitment.0, bits));
    }
    // Then: GTMul constraints for the reduction tree.
    let combine_mul_count = CombineDag::new(combine_commitments.len()).num_muls_total();
    for _ in 0..combine_mul_count {
        constraint_types.push(ConstraintType::GtMul);
    }

    let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false);
    let enable_g1_fused_wiring_end_to_end =
        std::env::var("JOLT_RECURSION_ENABLE_G1_FUSED_WIRING_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
    let enable_g1_scalar_mul_fused_end_to_end =
        std::env::var("JOLT_RECURSION_ENABLE_G1_SCALAR_MUL_FUSED_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
            || enable_g1_fused_wiring_end_to_end;
    let enable_g1_add_fused_end_to_end = enable_g1_fused_wiring_end_to_end;

    // Dense polynomial var count (needed for Hyrax setup bounds).
    let dense_num_vars = if enable_gt_fused_end_to_end
        || enable_g1_scalar_mul_fused_end_to_end
        || enable_g1_add_fused_end_to_end
    {
        PrefixPackingLayout::from_constraint_types_fused(
            &constraint_types,
            enable_gt_fused_end_to_end,
            enable_g1_scalar_mul_fused_end_to_end,
            enable_g1_add_fused_end_to_end,
        )
        .num_dense_vars
    } else {
        PrefixPackingLayout::from_constraint_types(&constraint_types).num_dense_vars
    };

    // Recursion verifier input expects matrix parameters derived from constraint counts.
    let num_constraints = constraint_types.len();
    let num_constraints_padded = num_constraints.next_power_of_two();
    let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
    let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
    let num_constraint_vars = 11usize;
    let num_vars = num_s_vars + num_constraint_vars;

    // Pairing boundary extraction from the final AssertEq constraint.
    let (lhs, rhs) = ast
        .constraints
        .iter()
        .find_map(|c| match c {
            AstConstraint::AssertEq { lhs, rhs, .. } => Some((*lhs, *rhs)),
        })
        .ok_or(ProofVerifyError::default())?;

    let lhs_idx = lhs.0 as usize;
    let rhs_idx = rhs.0 as usize;
    debug_assert!(
        lhs_idx < ast.nodes.len(),
        "lhs {} out of bounds (nodes len {})",
        lhs_idx,
        ast.nodes.len()
    );
    debug_assert!(
        rhs_idx < ast.nodes.len(),
        "rhs {} out of bounds (nodes len {})",
        rhs_idx,
        ast.nodes.len()
    );

    let (multi_id, rhs_id) = match &ast.nodes[lhs_idx].op {
        AstOp::MultiPairing { .. } => (lhs, rhs),
        _ => (rhs, lhs),
    };

    let multi_idx = multi_id.0 as usize;
    debug_assert!(
        multi_idx < ast.nodes.len(),
        "multi_id {} out of bounds",
        multi_idx
    );
    let (g1s, g2s) = match &ast.nodes[multi_idx].op {
        AstOp::MultiPairing { g1s, g2s, .. } => (g1s.clone(), g2s.clone()),
        _ => return Err(ProofVerifyError::default()),
    };
    if g1s.len() != 3 || g2s.len() != 3 {
        return Err(ProofVerifyError::default());
    }

    let eval_g1 = |id: ValueId| -> Result<G1Affine, ProofVerifyError> {
        let idx = id.0 as usize;
        debug_assert!(idx < values.len(), "eval_g1 id {} out of bounds", idx);
        match values[idx].as_ref().ok_or(ProofVerifyError::default())? {
            Value::G1(v) => Ok(v.0.into_affine()),
            _ => Err(ProofVerifyError::default()),
        }
    };
    let eval_g2 = |id: ValueId| -> Result<G2Affine, ProofVerifyError> {
        let idx = id.0 as usize;
        debug_assert!(idx < values.len(), "eval_g2 id {} out of bounds", idx);
        match values[idx].as_ref().ok_or(ProofVerifyError::default())? {
            Value::G2(v) => Ok(v.0.into_affine()),
            _ => Err(ProofVerifyError::default()),
        }
    };

    let rhs_val_idx = rhs_id.0 as usize;
    debug_assert!(
        rhs_val_idx < values.len(),
        "rhs_val_idx {} out of bounds",
        rhs_val_idx
    );
    let rhs_val: Fq12 = match values[rhs_val_idx]
        .as_ref()
        .ok_or(ProofVerifyError::default())?
    {
        Value::GT(v) => v.0,
        _ => return Err(ProofVerifyError::default()),
    };

    let pairing_boundary = PairingBoundary {
        p1_g1: eval_g1(g1s[0])?,
        p1_g2: eval_g2(g2s[0])?,
        p2_g1: eval_g1(g1s[1])?,
        p2_g2: eval_g2(g2s[1])?,
        p3_g1: eval_g1(g1s[2])?,
        p3_g2: eval_g2(g2s[2])?,
        rhs: rhs_val,
    };
    let wiring = derive_wiring_plan(ast, combine_commitments.len(), &pairing_boundary)?;

    Ok(DerivedRecursionInput {
        verifier_input: RecursionVerifierInput {
            constraint_types,
            enable_gt_fused_end_to_end,
            enable_g1_scalar_mul_fused_end_to_end,
            enable_g1_fused_wiring_end_to_end,
            num_vars,
            num_constraint_vars,
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            gt_exp_public_inputs,
            g1_scalar_mul_public_inputs,
            g2_scalar_mul_public_inputs,
            wiring,
            pairing_boundary: pairing_boundary.clone(),
            joint_commitment: joint_commitment.0,
        },
        dense_num_vars,
        pairing_boundary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::dory::witness::gt_exp::Base4ExponentiationSteps;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_bits_from_exponent_zero() {
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(0u64));
        assert!(bits.is_empty(), "zero exponent should produce empty bits");
    }

    #[test]
    fn test_bits_from_exponent_one() {
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(1u64));
        assert_eq!(bits, vec![true], "exponent 1 should produce [true]");
    }

    #[test]
    fn test_bits_from_exponent_two() {
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(2u64));
        assert_eq!(
            bits,
            vec![true, false],
            "exponent 2 should produce [true, false]"
        );
    }

    #[test]
    fn test_bits_from_exponent_three() {
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(3u64));
        assert_eq!(
            bits,
            vec![true, true],
            "exponent 3 should produce [true, true]"
        );
    }

    #[test]
    fn test_bits_from_exponent_four() {
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(4u64));
        assert_eq!(
            bits,
            vec![true, false, false],
            "exponent 4 should produce [true, false, false]"
        );
    }

    #[test]
    fn test_bits_from_exponent_large() {
        // 255 = 0b11111111
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(255u64));
        assert_eq!(bits.len(), 8, "exponent 255 should produce 8 bits");
        assert!(bits.iter().all(|&b| b), "all bits of 255 should be true");
    }

    #[test]
    fn test_bits_from_exponent_power_of_two() {
        // 256 = 2^8 = 0b100000000
        let bits = bits_from_exponent_msb_no_leading_zeros(Fr::from(256u64));
        assert_eq!(bits.len(), 9, "exponent 256 should produce 9 bits");
        assert!(bits[0], "first bit should be true");
        assert!(
            bits[1..].iter().all(|&b| !b),
            "remaining bits should be false"
        );
    }

    /// Verify that `bits_from_exponent_msb_no_leading_zeros` produces the same bits
    /// as `Base4ExponentiationSteps::new().bits` for various scalars.
    ///
    /// This is a critical consistency check: the verifier (instance_plan) and
    /// prover (witness generation) must agree on bit representation.
    #[test]
    fn test_bits_consistency_with_base4_exponentiation_steps() {
        let mut rng = thread_rng();
        let base = Fq12::from(42u64); // Arbitrary base, doesn't affect bit extraction

        // Test specific values
        for scalar_u64 in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 255, 256, 1000, 65535] {
            let scalar = Fr::from(scalar_u64);
            let plan_bits = bits_from_exponent_msb_no_leading_zeros(scalar);
            let witness_steps = Base4ExponentiationSteps::new(base, scalar);
            assert_eq!(
                plan_bits, witness_steps.bits,
                "bits mismatch for scalar {} (plan: {:?}, witness: {:?})",
                scalar_u64, plan_bits, witness_steps.bits
            );
        }

        // Test random values
        for _ in 0..10 {
            let scalar = Fr::rand(&mut rng);
            let plan_bits = bits_from_exponent_msb_no_leading_zeros(scalar);
            let witness_steps = Base4ExponentiationSteps::new(base, scalar);
            assert_eq!(
                plan_bits, witness_steps.bits,
                "bits mismatch for random scalar"
            );
        }
    }

    /// Test that bits conversion is deterministic
    #[test]
    fn test_bits_determinism() {
        let scalar = Fr::from(12345u64);
        let bits1 = bits_from_exponent_msb_no_leading_zeros(scalar);
        let bits2 = bits_from_exponent_msb_no_leading_zeros(scalar);
        assert_eq!(bits1, bits2, "bit conversion should be deterministic");
    }
}
