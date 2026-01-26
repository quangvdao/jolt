//! Derive canonical wiring/copy-constraint edge lists for recursion Stage 2.
//!
//! This module is the single source of truth for *what* is wired:
//! - Dory AST dataflow edges between proven operations (GT/G1/G2).
//! - Deterministic combine-commitments reduction DAG edges (`CombineDag`).
//! - Boundary bindings for `PairingBoundary` and `joint_commitment`.
//!
//! The actual wiring *sumcheck instances* live in per-family modules:
//! - `gt/wiring.rs`
//! - `g1/wiring.rs`
//! - `g2/wiring.rs`

use crate::utils::errors::ProofVerifyError;
use crate::zkvm::proof_serialization::PairingBoundary;
use crate::zkvm::recursion::CombineDag;
use dory::recursion::ast::{AstConstraint, AstGraph, AstOp, ValueId};
use dory::recursion::OpId;

/// Canonical wiring plan (verifier-derived, and mirrored by the prover).
#[derive(Clone, Debug, Default)]
pub struct WiringPlan {
    pub gt: Vec<GtWiringEdge>,
    pub g1: Vec<G1WiringEdge>,
    pub g2: Vec<G2WiringEdge>,
}

// =============================================================================
// GT wiring
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GtProducer {
    /// Output of a packed GT exponentiation: rho(s, x) (11-var).
    ///
    /// In the wiring polynomial, the endpoint selection is enforced by fixing the step bits
    /// in the Eq-kernel selector point; the producer is still `rho(s,x)` as a full 11-var MLE.
    GtExpRho { instance: usize },
    /// Output of a GT multiplication: result(x) (4-var).
    GtMulResult { instance: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GtConsumer {
    /// Input port 0 of a GT multiplication: lhs(x) (4-var).
    GtMulLhs { instance: usize },
    /// Input port 1 of a GT multiplication: rhs(x) (4-var).
    GtMulRhs { instance: usize },

    /// Boundary constant: base for the given GTExp instance.
    ///
    /// The constant value is taken from `RecursionVerifierInput.gt_exp_public_inputs[instance].base`.
    GtExpBase { instance: usize },

    /// Boundary constant: joint commitment used to build the Dory AST.
    JointCommitment,

    /// Boundary constant: external pairing check RHS.
    PairingBoundaryRhs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GtWiringEdge {
    pub src: GtProducer,
    pub dst: GtConsumer,
}

// =============================================================================
// G1 wiring
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum G1ValueRef {
    /// Output point of a G1 scalar mul: (x_a_next(s), y_a_next(s), a_indicator(s)).
    G1ScalarMulOut { instance: usize },
    /// Output point of a G1 add: (x_r, y_r, ind_r) (0-var).
    G1AddOut { instance: usize },

    /// Input P of a G1 add: (x_p, y_p, ind_p) (0-var).
    G1AddInP { instance: usize },
    /// Input Q of a G1 add: (x_q, y_q, ind_q) (0-var).
    G1AddInQ { instance: usize },

    /// Boundary constant: base point for the given G1 scalar mul constraint.
    ///
    /// The constant value is taken from `ConstraintType::G1ScalarMul { base_point }` (local index).
    G1ScalarMulBase { instance: usize },

    /// Boundary constant: external pairing check point (p1/p2/p3).
    PairingBoundaryP1,
    PairingBoundaryP2,
    PairingBoundaryP3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct G1WiringEdge {
    pub src: G1ValueRef,
    pub dst: G1ValueRef,
}

// =============================================================================
// G2 wiring
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum G2ValueRef {
    /// Output point of a G2 scalar mul: (x_a_next(s), y_a_next(s), a_indicator(s)).
    G2ScalarMulOut { instance: usize },
    /// Output point of a G2 add: (x_r, y_r, ind_r) (0-var).
    G2AddOut { instance: usize },

    /// Input P of a G2 add: (x_p, y_p, ind_p) (0-var).
    G2AddInP { instance: usize },
    /// Input Q of a G2 add: (x_q, y_q, ind_q) (0-var).
    G2AddInQ { instance: usize },

    /// Boundary constant: base point for the given G2 scalar mul constraint.
    ///
    /// The constant value is taken from `ConstraintType::G2ScalarMul { base_point }` (local index).
    G2ScalarMulBase { instance: usize },

    /// Boundary constant: external pairing check point (p1/p2/p3).
    PairingBoundaryP1,
    PairingBoundaryP2,
    PairingBoundaryP3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct G2WiringEdge {
    pub src: G2ValueRef,
    pub dst: G2ValueRef,
}

// =============================================================================
// Plan derivation
// =============================================================================

#[derive(Clone, Debug)]
struct OpIdOrder {
    gt_exp: Vec<OpId>,
    gt_mul: Vec<OpId>,
    g1_smul: Vec<OpId>,
    g2_smul: Vec<OpId>,
    g1_add: Vec<OpId>,
    g2_add: Vec<OpId>,
}

fn collect_op_ids(ast: &AstGraph<dory::backends::arkworks::BN254>) -> OpIdOrder {
    let mut out = OpIdOrder {
        gt_exp: Vec::new(),
        gt_mul: Vec::new(),
        g1_smul: Vec::new(),
        g2_smul: Vec::new(),
        g1_add: Vec::new(),
        g2_add: Vec::new(),
    };
    for node in &ast.nodes {
        match &node.op {
            AstOp::GTExp { op_id: Some(id), .. } => out.gt_exp.push(*id),
            AstOp::GTMul { op_id: Some(id), .. } => out.gt_mul.push(*id),
            AstOp::G1ScalarMul { op_id: Some(id), .. } => out.g1_smul.push(*id),
            AstOp::G2ScalarMul { op_id: Some(id), .. } => out.g2_smul.push(*id),
            AstOp::G1Add { op_id: Some(id), .. } => out.g1_add.push(*id),
            AstOp::G2Add { op_id: Some(id), .. } => out.g2_add.push(*id),
            _ => {}
        }
    }
    out.gt_exp.sort();
    out.gt_mul.sort();
    out.g1_smul.sort();
    out.g2_smul.sort();
    out.g1_add.sort();
    out.g2_add.sort();
    out
}

fn index_map(op_ids: &[OpId]) -> std::collections::BTreeMap<OpId, usize> {
    op_ids.iter().copied().enumerate().map(|(i, id)| (id, i)).collect()
}

fn gt_producer_from_value(
    ast: &AstGraph<dory::backends::arkworks::BN254>,
    gt_exp_index: &std::collections::BTreeMap<OpId, usize>,
    gt_mul_index: &std::collections::BTreeMap<OpId, usize>,
    value: ValueId,
) -> Option<GtProducer> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::GTExp { op_id: Some(id), .. } => gt_exp_index
            .get(id)
            .copied()
            .map(|instance| GtProducer::GtExpRho { instance }),
        AstOp::GTMul { op_id: Some(id), .. } => gt_mul_index
            .get(id)
            .copied()
            .map(|instance| GtProducer::GtMulResult { instance }),
        _ => None,
    }
}

fn g1_value_from_output(
    ast: &AstGraph<dory::backends::arkworks::BN254>,
    g1_smul_index: &std::collections::BTreeMap<OpId, usize>,
    g1_add_index: &std::collections::BTreeMap<OpId, usize>,
    value: ValueId,
) -> Option<G1ValueRef> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::G1ScalarMul { op_id: Some(id), .. } => {
            g1_smul_index.get(id).copied().map(|instance| G1ValueRef::G1ScalarMulOut { instance })
        }
        AstOp::G1Add { op_id: Some(id), .. } => {
            g1_add_index.get(id).copied().map(|instance| G1ValueRef::G1AddOut { instance })
        }
        _ => None,
    }
}

fn g2_value_from_output(
    ast: &AstGraph<dory::backends::arkworks::BN254>,
    g2_smul_index: &std::collections::BTreeMap<OpId, usize>,
    g2_add_index: &std::collections::BTreeMap<OpId, usize>,
    value: ValueId,
) -> Option<G2ValueRef> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::G2ScalarMul { op_id: Some(id), .. } => {
            g2_smul_index.get(id).copied().map(|instance| G2ValueRef::G2ScalarMulOut { instance })
        }
        AstOp::G2Add { op_id: Some(id), .. } => {
            g2_add_index.get(id).copied().map(|instance| G2ValueRef::G2AddOut { instance })
        }
        _ => None,
    }
}

/// Derive the full wiring plan for Stage 2 wiring/boundary constraints.
///
/// Inputs required for deterministic combine wiring:
/// - `combine_leaves`: number of commitments combined in Stage 8 (leaves of the deterministic DAG)
/// - `joint_commitment`: the GT value used as the Dory AST \"commitment\" input
/// - `pairing_boundary`: boundary outputs to bind (p1/p2/p3,rhs)
///
/// Returns: canonical edge lists (GT/G1/G2).
pub fn derive_wiring_plan(
    ast: &AstGraph<dory::backends::arkworks::BN254>,
    combine_leaves: usize,
    _pairing_boundary: &PairingBoundary,
) -> Result<WiringPlan, ProofVerifyError> {
    let order = collect_op_ids(ast);
    let gt_exp_index = index_map(&order.gt_exp);
    let gt_mul_index = index_map(&order.gt_mul);
    let g1_smul_index = index_map(&order.g1_smul);
    let g2_smul_index = index_map(&order.g2_smul);
    let g1_add_index = index_map(&order.g1_add);
    let g2_add_index = index_map(&order.g2_add);

    let mut plan = WiringPlan::default();

    // --- AST internal dataflow edges (copy constraints) ---
    for node in &ast.nodes {
        match &node.op {
            AstOp::GTMul {
                op_id: Some(id),
                lhs,
                rhs,
                ..
            } => {
                let Some(&mul_instance) = gt_mul_index.get(id) else { continue };
                if let Some(src) = gt_producer_from_value(ast, &gt_exp_index, &gt_mul_index, *lhs) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtMulLhs {
                            instance: mul_instance,
                        },
                    });
                }
                if let Some(src) = gt_producer_from_value(ast, &gt_exp_index, &gt_mul_index, *rhs) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtMulRhs {
                            instance: mul_instance,
                        },
                    });
                }
            }
            AstOp::G1Add {
                op_id: Some(id),
                a,
                b,
                ..
            } => {
                let Some(&add_instance) = g1_add_index.get(id) else { continue };
                if let Some(src) = g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *a) {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1AddInP {
                            instance: add_instance,
                        },
                    });
                }
                if let Some(src) = g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *b) {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1AddInQ {
                            instance: add_instance,
                        },
                    });
                }
            }
            AstOp::G2Add {
                op_id: Some(id),
                a,
                b,
                ..
            } => {
                let Some(&add_instance) = g2_add_index.get(id) else { continue };
                if let Some(src) = g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *a) {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2AddInP {
                            instance: add_instance,
                        },
                    });
                }
                if let Some(src) = g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *b) {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2AddInQ {
                            instance: add_instance,
                        },
                    });
                }
            }
            _ => {}
        }
    }

    // --- Base/point binding edges (bind non-input hints) ---
    for node in &ast.nodes {
        match &node.op {
            AstOp::GTExp {
                op_id: Some(id),
                base,
                ..
            } => {
                let Some(&exp_instance) = gt_exp_index.get(id) else { continue };
                if let Some(src) = gt_producer_from_value(ast, &gt_exp_index, &gt_mul_index, *base) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtExpBase {
                            instance: exp_instance,
                        },
                    });
                }
            }
            AstOp::G1ScalarMul {
                op_id: Some(id),
                point,
                ..
            } => {
                let Some(&smul_instance) = g1_smul_index.get(id) else { continue };
                if let Some(src) =
                    g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *point)
                {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1ScalarMulBase {
                            instance: smul_instance,
                        },
                    });
                }
            }
            AstOp::G2ScalarMul {
                op_id: Some(id),
                point,
                ..
            } => {
                let Some(&smul_instance) = g2_smul_index.get(id) else { continue };
                if let Some(src) =
                    g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *point)
                {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2ScalarMulBase {
                            instance: smul_instance,
                        },
                    });
                }
            }
            _ => {}
        }
    }

    // --- Pairing boundary bindings (p1/p2/p3,rhs) ---
    //
    // Extract MultiPairing node and RHS value id from the final AssertEq constraint.
    let (lhs, rhs) = ast
        .constraints
        .iter()
        .find_map(|c| match c {
            AstConstraint::AssertEq { lhs, rhs, .. } => Some((*lhs, *rhs)),
        })
        .ok_or(ProofVerifyError::default())?;

    let lhs_idx = lhs.0 as usize;
    let rhs_idx = rhs.0 as usize;
    if lhs_idx >= ast.nodes.len() || rhs_idx >= ast.nodes.len() {
        return Err(ProofVerifyError::default());
    }
    let (multi_id, rhs_id) = match &ast.nodes[lhs_idx].op {
        AstOp::MultiPairing { .. } => (lhs, rhs),
        _ => (rhs, lhs),
    };
    let multi_idx = multi_id.0 as usize;
    if multi_idx >= ast.nodes.len() {
        return Err(ProofVerifyError::default());
    }
    let (g1s, g2s) = match &ast.nodes[multi_idx].op {
        AstOp::MultiPairing { g1s, g2s, .. } => (g1s.clone(), g2s.clone()),
        _ => return Err(ProofVerifyError::default()),
    };
    if g1s.len() != 3 || g2s.len() != 3 {
        return Err(ProofVerifyError::default());
    }

    // Bind G1 pairing inputs.
    for (i, vid) in g1s.iter().enumerate() {
        let Some(src) = g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *vid) else {
            continue;
        };
        let dst = match i {
            0 => G1ValueRef::PairingBoundaryP1,
            1 => G1ValueRef::PairingBoundaryP2,
            2 => G1ValueRef::PairingBoundaryP3,
            _ => unreachable!(),
        };
        plan.g1.push(G1WiringEdge { src, dst });
    }

    // Bind G2 pairing inputs.
    for (i, vid) in g2s.iter().enumerate() {
        let Some(src) = g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *vid) else {
            continue;
        };
        let dst = match i {
            0 => G2ValueRef::PairingBoundaryP1,
            1 => G2ValueRef::PairingBoundaryP2,
            2 => G2ValueRef::PairingBoundaryP3,
            _ => unreachable!(),
        };
        plan.g2.push(G2WiringEdge { src, dst });
    }

    // Bind pairing RHS (GT).
    if let Some(src) = gt_producer_from_value(ast, &gt_exp_index, &gt_mul_index, rhs_id) {
        plan.gt.push(GtWiringEdge {
            src,
            dst: GtConsumer::PairingBoundaryRhs,
        });
    }

    // --- Combine-commitments wiring (GT) + binding to joint commitment ---
    //
    // Combine constraints are appended *after* Dory-traced constraints:
    // - first: `combine_leaves` GTExp instances
    // - then:  `combine_leaves-1` GTMul instances in deterministic balanced-fold order.
    if combine_leaves > 0 {
        let dory_gt_exp = order.gt_exp.len();
        let dory_gt_mul = order.gt_mul.len();
        let combine_exp_start = dory_gt_exp;
        let combine_mul_start = dory_gt_mul;
        let expected_mul_count = CombineDag::new(combine_leaves).num_muls_total();

        let mut nodes: Vec<GtProducer> = (0..combine_leaves)
            .map(|i| GtProducer::GtExpRho {
                instance: combine_exp_start + i,
            })
            .collect();
        let mut mul_idx = 0usize;
        while nodes.len() > 1 {
            let mut next: Vec<GtProducer> = Vec::with_capacity((nodes.len() + 1) / 2);
            for chunk in nodes.chunks(2) {
                if let [a, b] = chunk {
                    let inst = combine_mul_start + mul_idx;
                    mul_idx += 1;
                    plan.gt.push(GtWiringEdge {
                        src: *a,
                        dst: GtConsumer::GtMulLhs { instance: inst },
                    });
                    plan.gt.push(GtWiringEdge {
                        src: *b,
                        dst: GtConsumer::GtMulRhs { instance: inst },
                    });
                    next.push(GtProducer::GtMulResult { instance: inst });
                } else {
                    next.push(chunk[0]);
                }
            }
            nodes = next;
        }
        debug_assert_eq!(
            mul_idx, expected_mul_count,
            "combine wiring mul count mismatch"
        );

        // Bind combine root output to the joint commitment used by the Dory AST.
        //
        // The *value* of the joint commitment is provided separately (in `RecursionVerifierInput`),
        // and the wiring verifier checks equality against that constant.
        let root = nodes[0];
        plan.gt.push(GtWiringEdge {
            src: root,
            dst: GtConsumer::JointCommitment,
        });
    }

    // Canonical edge ordering (must match prover and verifier): stable sort by dst then src.
    plan.gt.sort_by_key(|e| (e.dst, e.src));
    plan.g1.sort_by_key(|e| (e.dst, e.src));
    plan.g2.sort_by_key(|e| (e.dst, e.src));

    Ok(plan)
}

