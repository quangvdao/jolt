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

use std::io::{ErrorKind, Read, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use dory::backends::BN254;
use dory::recursion::ast::{AstConstraint, AstGraph, AstOp, InputSource, ValueId};
use dory::recursion::OpId;

use crate::utils::errors::ProofVerifyError;
use crate::zkvm::proof_serialization::PairingBoundary;
use crate::zkvm::{guest_serde::GuestDeserialize, guest_serde::GuestSerialize};

use super::constraints::system::ConstraintType;
use super::CombineDag;

/// Canonical wiring plan (verifier-derived, and mirrored by the prover).
#[derive(Clone, Debug, Default)]
pub struct WiringPlan {
    pub gt: Vec<GtWiringEdge>,
    pub g1: Vec<G1WiringEdge>,
    pub g2: Vec<G2WiringEdge>,
}

impl CanonicalSerialize for WiringPlan {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.gt.serialize_with_mode(&mut writer, compress)?;
        self.g1.serialize_with_mode(&mut writer, compress)?;
        self.g2.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.gt.serialized_size(compress)
            + self.g1.serialized_size(compress)
            + self.g2.serialized_size(compress)
    }
}

impl Valid for WiringPlan {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for WiringPlan {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            gt: Vec::<GtWiringEdge>::deserialize_with_mode(&mut reader, compress, validate)?,
            g1: Vec::<G1WiringEdge>::deserialize_with_mode(&mut reader, compress, validate)?,
            g2: Vec::<G2WiringEdge>::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

impl GuestSerialize for WiringPlan {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.gt.guest_serialize(w)?;
        self.g1.guest_serialize(w)?;
        self.g2.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for WiringPlan {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            gt: Vec::<GtWiringEdge>::guest_deserialize(r)?,
            g1: Vec::<G1WiringEdge>::guest_deserialize(r)?,
            g2: Vec::<G2WiringEdge>::guest_deserialize(r)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GtProducer {
    /// Output of a packed GT exponentiation: rho(s, x) (11-var).
    ///
    /// In the wiring polynomial, the endpoint selection is enforced by fixing the step bits
    /// in the Eq-kernel selector point; the producer is still `rho(s,x)` as a full 11-var MLE.
    GtExpRho { instance: usize },
    /// Boundary constant: base for the given GTExp instance.
    ///
    /// This is used when the Dory AST wires an *input* GT value into another port (e.g. GTMul
    /// lhs/rhs). In that case we treat the input as the GTExp-base constant for that instance,
    /// whose value is available to the verifier via `RecursionVerifierInput.gt_exp_public_inputs`.
    GtExpBase { instance: usize },
    /// Output of a GT multiplication: result(x) (4-var).
    GtMulResult { instance: usize },
    /// Output of a BN254 Miller loop (pre-final-exponentiation) for a specific pairing instance.
    ///
    /// The `instance` is the **global constraint index** of the `ConstraintType::MultiMillerLoop`
    /// constraint. This matches the `ConstraintListProver` virtual polynomial identifiers.
    MultiMillerLoopOut { instance: usize },
    /// Boundary constant: Stage-8 joint commitment (GT).
    ///
    /// This appears as an `AstOp::Input` with `InputSource::Proof { name: "commitment" }` and can
    /// be wired directly into GTMul ports in small Dory ASTs.
    JointCommitment,
    /// Boundary constant: a GT-valued `AstOp::Input` that is consumed directly by a GT port
    /// (e.g. GTMul lhs/rhs) but is **not** represented as a GTExp base boundary.
    ///
    /// The value is resolved deterministically from the Dory proof / setup on both prover and
    /// verifier sides, keyed by the AST node `ValueId` index.
    GtInput { value_id: u32 },
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
    /// Boundary constant: pairing Miller loop RHS (pre-final-exponentiation).
    PairingBoundaryMillerRhs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GtWiringEdge {
    pub src: GtProducer,
    pub dst: GtConsumer,
}

impl CanonicalSerialize for GtWiringEdge {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.src.serialize_with_mode(&mut writer, compress)?;
        self.dst.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.src.serialized_size(compress) + self.dst.serialized_size(compress)
    }
}

impl Valid for GtWiringEdge {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for GtWiringEdge {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            src: GtProducer::deserialize_with_mode(&mut reader, compress, validate)?,
            dst: GtConsumer::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

impl GuestSerialize for GtWiringEdge {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.src.guest_serialize(w)?;
        self.dst.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for GtWiringEdge {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            src: GtProducer::guest_deserialize(r)?,
            dst: GtConsumer::guest_deserialize(r)?,
        })
    }
}

impl CanonicalSerialize for GtProducer {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            GtProducer::GtExpRho { instance } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtProducer::GtMulResult { instance } => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtProducer::GtExpBase { instance } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtProducer::MultiMillerLoopOut { instance } => {
                5u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtProducer::JointCommitment => {
                // Keep encoding size uniform: tag + dummy u32.
                3u8.serialize_with_mode(&mut writer, compress)?;
                0u32.serialize_with_mode(&mut writer, compress)
            }
            GtProducer::GtInput { value_id } => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                value_id.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        // tag (u8) + instance (u32)
        1 + 4
    }
}

impl Valid for GtProducer {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for GtProducer {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(match tag {
            0 => Self::GtExpRho {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            1 => Self::GtMulResult {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            2 => Self::GtExpBase {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            3 => {
                let _ = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                Self::JointCommitment
            }
            4 => Self::GtInput {
                value_id: u32::deserialize_with_mode(&mut reader, compress, validate)?,
            },
            5 => Self::MultiMillerLoopOut {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for GtProducer {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            GtProducer::GtExpRho { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtProducer instance overflow")
                })?)
                .guest_serialize(w)
            }
            GtProducer::GtMulResult { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtProducer instance overflow")
                })?)
                .guest_serialize(w)
            }
            GtProducer::GtExpBase { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtProducer instance overflow")
                })?)
                .guest_serialize(w)
            }
            GtProducer::JointCommitment => {
                3u8.guest_serialize(w)?;
                0u32.guest_serialize(w)
            }
            GtProducer::GtInput { value_id } => {
                4u8.guest_serialize(w)?;
                value_id.guest_serialize(w)
            }
            GtProducer::MultiMillerLoopOut { instance } => {
                5u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtProducer instance overflow")
                })?)
                .guest_serialize(w)
            }
        }
    }
}

impl GuestDeserialize for GtProducer {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let tag = u8::guest_deserialize(r)?;
        Ok(match tag {
            0 => Self::GtExpRho {
                instance: u32::guest_deserialize(r)? as usize,
            },
            1 => Self::GtMulResult {
                instance: u32::guest_deserialize(r)? as usize,
            },
            2 => Self::GtExpBase {
                instance: u32::guest_deserialize(r)? as usize,
            },
            3 => {
                let _ = u32::guest_deserialize(r)?;
                Self::JointCommitment
            }
            4 => Self::GtInput {
                value_id: u32::guest_deserialize(r)?,
            },
            5 => Self::MultiMillerLoopOut {
                instance: u32::guest_deserialize(r)? as usize,
            },
            _ => {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "invalid GtProducer tag",
                ))
            }
        })
    }
}

impl CanonicalSerialize for GtConsumer {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            GtConsumer::GtMulLhs { instance } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtConsumer::GtMulRhs { instance } => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtConsumer::GtExpBase { instance } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            GtConsumer::JointCommitment => 3u8.serialize_with_mode(&mut writer, compress),
            GtConsumer::PairingBoundaryRhs => 4u8.serialize_with_mode(&mut writer, compress),
            GtConsumer::PairingBoundaryMillerRhs => 5u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        match self {
            GtConsumer::GtMulLhs { .. }
            | GtConsumer::GtMulRhs { .. }
            | GtConsumer::GtExpBase { .. } => 1 + 4,
            GtConsumer::JointCommitment
            | GtConsumer::PairingBoundaryRhs
            | GtConsumer::PairingBoundaryMillerRhs => 1,
        }
    }
}

impl Valid for GtConsumer {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for GtConsumer {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(match tag {
            0 => Self::GtMulLhs {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            1 => Self::GtMulRhs {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            2 => Self::GtExpBase {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            3 => Self::JointCommitment,
            4 => Self::PairingBoundaryRhs,
            5 => Self::PairingBoundaryMillerRhs,
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for GtConsumer {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            GtConsumer::GtMulLhs { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtConsumer instance overflow")
                })?)
                .guest_serialize(w)
            }
            GtConsumer::GtMulRhs { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtConsumer instance overflow")
                })?)
                .guest_serialize(w)
            }
            GtConsumer::GtExpBase { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "GtConsumer instance overflow")
                })?)
                .guest_serialize(w)
            }
            GtConsumer::JointCommitment => 3u8.guest_serialize(w),
            GtConsumer::PairingBoundaryRhs => 4u8.guest_serialize(w),
            GtConsumer::PairingBoundaryMillerRhs => 5u8.guest_serialize(w),
        }
    }
}

impl GuestDeserialize for GtConsumer {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let tag = u8::guest_deserialize(r)?;
        Ok(match tag {
            0 => Self::GtMulLhs {
                instance: u32::guest_deserialize(r)? as usize,
            },
            1 => Self::GtMulRhs {
                instance: u32::guest_deserialize(r)? as usize,
            },
            2 => Self::GtExpBase {
                instance: u32::guest_deserialize(r)? as usize,
            },
            3 => Self::JointCommitment,
            4 => Self::PairingBoundaryRhs,
            5 => Self::PairingBoundaryMillerRhs,
            _ => {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "invalid GtConsumer tag",
                ))
            }
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum G1ValueRef {
    /// Output point of a G1 scalar mul: (x_a_next(s), y_a_next(s), a_indicator(s)).
    G1ScalarMulOut {
        instance: usize,
    },
    /// Output point of a G1 add: (x_r, y_r, ind_r) (0-var).
    G1AddOut {
        instance: usize,
    },

    /// Input P of a G1 add: (x_p, y_p, ind_p) (0-var).
    G1AddInP {
        instance: usize,
    },
    /// Input Q of a G1 add: (x_q, y_q, ind_q) (0-var).
    G1AddInQ {
        instance: usize,
    },

    /// Boundary constant: base point for the given G1 scalar mul constraint.
    ///
    /// This value is committed as part of the G1 scalar-mul witness rows (as c-only base
    /// polynomials). Wiring constraints bind it to the AST producer of the scalar-mul point input.
    G1ScalarMulBase {
        instance: usize,
    },

    /// Boundary constant for a G1 scalar-mul base when the point is an `AstOp::Input`.
    ///
    /// The constant value is taken from `ConstraintType::G1ScalarMul { base_point }` in local
    /// scalar-mul instance order (OpId-sorted), and is used to bind the committed base rows to the
    /// public Dory input value.
    G1ScalarMulBaseBoundary {
        instance: usize,
    },

    /// Boundary constant: a G1-valued `AstOp::Input` that feeds directly into a G1 port
    /// (e.g. G1Add inputs).
    ///
    /// The value is resolved deterministically from the Dory proof / setup, keyed by the AST node
    /// `ValueId` index.
    G1Input {
        value_id: u32,
    },

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

impl CanonicalSerialize for G1WiringEdge {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.src.serialize_with_mode(&mut writer, compress)?;
        self.dst.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.src.serialized_size(compress) + self.dst.serialized_size(compress)
    }
}

impl Valid for G1WiringEdge {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for G1WiringEdge {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            src: G1ValueRef::deserialize_with_mode(&mut reader, compress, validate)?,
            dst: G1ValueRef::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

impl GuestSerialize for G1WiringEdge {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.src.guest_serialize(w)?;
        self.dst.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for G1WiringEdge {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            src: G1ValueRef::guest_deserialize(r)?,
            dst: G1ValueRef::guest_deserialize(r)?,
        })
    }
}

impl CanonicalSerialize for G1ValueRef {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            G1ValueRef::G1ScalarMulOut { instance } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G1ValueRef::G1AddOut { instance } => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G1ValueRef::G1AddInP { instance } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G1ValueRef::G1AddInQ { instance } => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G1ValueRef::G1ScalarMulBase { instance } => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G1ValueRef::PairingBoundaryP1 => 5u8.serialize_with_mode(&mut writer, compress),
            G1ValueRef::PairingBoundaryP2 => 6u8.serialize_with_mode(&mut writer, compress),
            G1ValueRef::PairingBoundaryP3 => 7u8.serialize_with_mode(&mut writer, compress),
            G1ValueRef::G1ScalarMulBaseBoundary { instance } => {
                8u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G1ValueRef::G1Input { value_id } => {
                9u8.serialize_with_mode(&mut writer, compress)?;
                value_id.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        match self {
            G1ValueRef::G1ScalarMulOut { .. }
            | G1ValueRef::G1AddOut { .. }
            | G1ValueRef::G1AddInP { .. }
            | G1ValueRef::G1AddInQ { .. }
            | G1ValueRef::G1ScalarMulBase { .. }
            | G1ValueRef::G1ScalarMulBaseBoundary { .. }
            | G1ValueRef::G1Input { .. } => 1 + 4,
            G1ValueRef::PairingBoundaryP1
            | G1ValueRef::PairingBoundaryP2
            | G1ValueRef::PairingBoundaryP3 => 1,
        }
    }
}

impl Valid for G1ValueRef {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for G1ValueRef {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(match tag {
            0 => Self::G1ScalarMulOut {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            1 => Self::G1AddOut {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            2 => Self::G1AddInP {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            3 => Self::G1AddInQ {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            4 => Self::G1ScalarMulBase {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            5 => Self::PairingBoundaryP1,
            6 => Self::PairingBoundaryP2,
            7 => Self::PairingBoundaryP3,
            8 => Self::G1ScalarMulBaseBoundary {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            9 => Self::G1Input {
                value_id: u32::deserialize_with_mode(&mut reader, compress, validate)?,
            },
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for G1ValueRef {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            G1ValueRef::G1ScalarMulOut { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G1ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1AddOut { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G1ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1AddInP { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G1ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1AddInQ { instance } => {
                3u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G1ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1ScalarMulBase { instance } => {
                4u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G1ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::PairingBoundaryP1 => 5u8.guest_serialize(w),
            G1ValueRef::PairingBoundaryP2 => 6u8.guest_serialize(w),
            G1ValueRef::PairingBoundaryP3 => 7u8.guest_serialize(w),
            G1ValueRef::G1ScalarMulBaseBoundary { instance } => {
                8u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G1ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1Input { value_id } => {
                9u8.guest_serialize(w)?;
                value_id.guest_serialize(w)
            }
        }
    }
}

impl GuestDeserialize for G1ValueRef {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let tag = u8::guest_deserialize(r)?;
        Ok(match tag {
            0 => Self::G1ScalarMulOut {
                instance: u32::guest_deserialize(r)? as usize,
            },
            1 => Self::G1AddOut {
                instance: u32::guest_deserialize(r)? as usize,
            },
            2 => Self::G1AddInP {
                instance: u32::guest_deserialize(r)? as usize,
            },
            3 => Self::G1AddInQ {
                instance: u32::guest_deserialize(r)? as usize,
            },
            4 => Self::G1ScalarMulBase {
                instance: u32::guest_deserialize(r)? as usize,
            },
            5 => Self::PairingBoundaryP1,
            6 => Self::PairingBoundaryP2,
            7 => Self::PairingBoundaryP3,
            8 => Self::G1ScalarMulBaseBoundary {
                instance: u32::guest_deserialize(r)? as usize,
            },
            9 => Self::G1Input {
                value_id: u32::guest_deserialize(r)?,
            },
            _ => {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "invalid G1ValueRef tag",
                ))
            }
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum G2ValueRef {
    /// Output point of a G2 scalar mul: (x_a_next(s), y_a_next(s), a_indicator(s)).
    G2ScalarMulOut {
        instance: usize,
    },
    /// Output point of a G2 add: (x_r, y_r, ind_r) (0-var).
    G2AddOut {
        instance: usize,
    },

    /// Input P of a G2 add: (x_p, y_p, ind_p) (0-var).
    G2AddInP {
        instance: usize,
    },
    /// Input Q of a G2 add: (x_q, y_q, ind_q) (0-var).
    G2AddInQ {
        instance: usize,
    },

    /// Boundary constant: base point for the given G2 scalar mul constraint.
    ///
    /// This value is committed as part of the G2 scalar-mul witness rows (as step-constant base
    /// polynomials). Wiring constraints bind it to the AST producer of the scalar-mul point input.
    G2ScalarMulBase {
        instance: usize,
    },

    /// Boundary constant for a G2 scalar-mul base when the point is an `AstOp::Input`.
    ///
    /// The constant value is taken from `ConstraintType::G2ScalarMul { base_point }` in local
    /// scalar-mul instance order (OpId-sorted), and is used to bind the committed base rows to the
    /// public Dory input value.
    G2ScalarMulBaseBoundary {
        instance: usize,
    },

    /// Boundary constant: a G2-valued `AstOp::Input` that feeds directly into a G2 port
    /// (e.g. G2Add inputs).
    ///
    /// The value is resolved deterministically from the Dory proof / setup, keyed by the AST node
    /// `ValueId` index.
    G2Input {
        value_id: u32,
    },

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

impl CanonicalSerialize for G2WiringEdge {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.src.serialize_with_mode(&mut writer, compress)?;
        self.dst.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.src.serialized_size(compress) + self.dst.serialized_size(compress)
    }
}

impl Valid for G2WiringEdge {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for G2WiringEdge {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            src: G2ValueRef::deserialize_with_mode(&mut reader, compress, validate)?,
            dst: G2ValueRef::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

impl GuestSerialize for G2WiringEdge {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.src.guest_serialize(w)?;
        self.dst.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for G2WiringEdge {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            src: G2ValueRef::guest_deserialize(r)?,
            dst: G2ValueRef::guest_deserialize(r)?,
        })
    }
}

impl CanonicalSerialize for G2ValueRef {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            G2ValueRef::G2ScalarMulOut { instance } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::G2AddOut { instance } => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::G2AddInP { instance } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::G2AddInQ { instance } => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::G2ScalarMulBase { instance } => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::G2ScalarMulBaseBoundary { instance } => {
                5u8.serialize_with_mode(&mut writer, compress)?;
                (*instance as u32).serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::G2Input { value_id } => {
                9u8.serialize_with_mode(&mut writer, compress)?;
                value_id.serialize_with_mode(&mut writer, compress)
            }
            G2ValueRef::PairingBoundaryP1 => 6u8.serialize_with_mode(&mut writer, compress),
            G2ValueRef::PairingBoundaryP2 => 7u8.serialize_with_mode(&mut writer, compress),
            G2ValueRef::PairingBoundaryP3 => 8u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        match self {
            G2ValueRef::G2ScalarMulOut { .. }
            | G2ValueRef::G2AddOut { .. }
            | G2ValueRef::G2AddInP { .. }
            | G2ValueRef::G2AddInQ { .. }
            | G2ValueRef::G2ScalarMulBase { .. }
            | G2ValueRef::G2ScalarMulBaseBoundary { .. }
            | G2ValueRef::G2Input { .. } => 1 + 4,
            G2ValueRef::PairingBoundaryP1
            | G2ValueRef::PairingBoundaryP2
            | G2ValueRef::PairingBoundaryP3 => 1,
        }
    }
}

impl Valid for G2ValueRef {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for G2ValueRef {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(match tag {
            0 => Self::G2ScalarMulOut {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            1 => Self::G2AddOut {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            2 => Self::G2AddInP {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            3 => Self::G2AddInQ {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            4 => Self::G2ScalarMulBase {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            5 => Self::G2ScalarMulBaseBoundary {
                instance: u32::deserialize_with_mode(&mut reader, compress, validate)? as usize,
            },
            6 => Self::PairingBoundaryP1,
            7 => Self::PairingBoundaryP2,
            8 => Self::PairingBoundaryP3,
            9 => Self::G2Input {
                value_id: u32::deserialize_with_mode(&mut reader, compress, validate)?,
            },
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for G2ValueRef {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            G2ValueRef::G2ScalarMulOut { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G2ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2AddOut { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G2ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2AddInP { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G2ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2AddInQ { instance } => {
                3u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G2ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2ScalarMulBase { instance } => {
                4u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G2ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2ScalarMulBaseBoundary { instance } => {
                5u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(ErrorKind::InvalidData, "G2ValueRef instance overflow")
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2Input { value_id } => {
                9u8.guest_serialize(w)?;
                value_id.guest_serialize(w)
            }
            G2ValueRef::PairingBoundaryP1 => 6u8.guest_serialize(w),
            G2ValueRef::PairingBoundaryP2 => 7u8.guest_serialize(w),
            G2ValueRef::PairingBoundaryP3 => 8u8.guest_serialize(w),
        }
    }
}

impl GuestDeserialize for G2ValueRef {
    fn guest_deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let tag = u8::guest_deserialize(r)?;
        Ok(match tag {
            0 => Self::G2ScalarMulOut {
                instance: u32::guest_deserialize(r)? as usize,
            },
            1 => Self::G2AddOut {
                instance: u32::guest_deserialize(r)? as usize,
            },
            2 => Self::G2AddInP {
                instance: u32::guest_deserialize(r)? as usize,
            },
            3 => Self::G2AddInQ {
                instance: u32::guest_deserialize(r)? as usize,
            },
            4 => Self::G2ScalarMulBase {
                instance: u32::guest_deserialize(r)? as usize,
            },
            5 => Self::G2ScalarMulBaseBoundary {
                instance: u32::guest_deserialize(r)? as usize,
            },
            6 => Self::PairingBoundaryP1,
            7 => Self::PairingBoundaryP2,
            8 => Self::PairingBoundaryP3,
            9 => Self::G2Input {
                value_id: u32::guest_deserialize(r)?,
            },
            _ => {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "invalid G2ValueRef tag",
                ))
            }
        })
    }
}

fn gt_producer_from_value(
    ast: &AstGraph<BN254>,
    gt_exp_out_instance_by_value: &[Option<usize>],
    gt_mul_out_instance_by_value: &[Option<usize>],
    gt_exp_base_instance_by_value: &[Option<usize>],
    value: ValueId,
) -> Option<GtProducer> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::GTExp { .. } => gt_exp_out_instance_by_value
            .get(idx)
            .copied()
            .flatten()
            .map(|instance| GtProducer::GtExpRho { instance }),
        AstOp::GTMul { .. } => gt_mul_out_instance_by_value
            .get(idx)
            .copied()
            .flatten()
            .map(|instance| GtProducer::GtMulResult { instance }),
        AstOp::Input { source } => {
            if let Some(instance) = gt_exp_base_instance_by_value.get(idx).copied().flatten() {
                Some(GtProducer::GtExpBase { instance })
            } else {
                match source {
                    InputSource::Proof { name } if *name == "commitment" => {
                        Some(GtProducer::JointCommitment)
                    }
                    _ => Some(GtProducer::GtInput { value_id: value.0 }),
                }
            }
        }
        _ => None,
    }
}

fn is_ast_input(ast: &AstGraph<BN254>, value: ValueId) -> bool {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return false;
    }
    matches!(&ast.nodes[idx].op, AstOp::Input { .. })
}

fn g1_value_from_output(
    ast: &AstGraph<BN254>,
    g1_smul_out_instance_by_value: &[Option<usize>],
    g1_add_out_instance_by_value: &[Option<usize>],
    value: ValueId,
) -> Option<G1ValueRef> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::G1ScalarMul { .. } => g1_smul_out_instance_by_value
            .get(idx)
            .copied()
            .flatten()
            .map(|instance| G1ValueRef::G1ScalarMulOut { instance }),
        AstOp::G1Add { .. } => g1_add_out_instance_by_value
            .get(idx)
            .copied()
            .flatten()
            .map(|instance| G1ValueRef::G1AddOut { instance }),
        _ => None,
    }
}

fn g2_value_from_output(
    ast: &AstGraph<BN254>,
    g2_smul_out_instance_by_value: &[Option<usize>],
    g2_add_out_instance_by_value: &[Option<usize>],
    value: ValueId,
) -> Option<G2ValueRef> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::G2ScalarMul { .. } => g2_smul_out_instance_by_value
            .get(idx)
            .copied()
            .flatten()
            .map(|instance| G2ValueRef::G2ScalarMulOut { instance }),
        AstOp::G2Add { .. } => g2_add_out_instance_by_value
            .get(idx)
            .copied()
            .flatten()
            .map(|instance| G2ValueRef::G2AddOut { instance }),
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
    ast: &AstGraph<BN254>,
    combine_leaves: usize,
    _pairing_boundary: &PairingBoundary,
    constraint_types: &[ConstraintType],
) -> Result<WiringPlan, ProofVerifyError> {
    let n = ast.nodes.len();
    let mut gt_exp_out_instance_by_value: Vec<Option<usize>> = vec![None; n];
    let mut gt_mul_out_instance_by_value: Vec<Option<usize>> = vec![None; n];
    let mut g1_smul_out_instance_by_value: Vec<Option<usize>> = vec![None; n];
    let mut g2_smul_out_instance_by_value: Vec<Option<usize>> = vec![None; n];
    let mut g1_add_out_instance_by_value: Vec<Option<usize>> = vec![None; n];
    let mut g2_add_out_instance_by_value: Vec<Option<usize>> = vec![None; n];
    // Map ValueId(Input GT value) -> GTExp instance index whose base is that ValueId.
    // This lets us treat GT inputs as the corresponding GTExp base constant in wiring.
    let mut gt_exp_base_instance_by_value: Vec<Option<usize>> = vec![None; n];

    // Pass 1: assign per-type instance indices and populate lookup tables.
    //
    // IMPORTANT: We rely on Dory's `OpIdBuilder` monotonicity to ensure the encounter order in
    // `ast.nodes` matches the implicit instance ordering used by the recursion extractor/prover.
    // Keep debug-only assertions as a safety net.
    let mut dory_gt_exp = 0usize;
    let mut dory_gt_mul = 0usize;
    let mut dory_g1_smul = 0usize;
    let mut dory_g2_smul = 0usize;
    let mut dory_g1_add = 0usize;
    let mut dory_g2_add = 0usize;

    let mut last_gt_exp: Option<OpId> = None;
    let mut last_gt_mul: Option<OpId> = None;
    let mut last_g1_smul: Option<OpId> = None;
    let mut last_g2_smul: Option<OpId> = None;
    let mut last_g1_add: Option<OpId> = None;
    let mut last_g2_add: Option<OpId> = None;

    for node in &ast.nodes {
        let out_idx = node.out.0 as usize;
        debug_assert!(
            out_idx < n,
            "AstGraph invariant violated: out out of bounds"
        );
        match &node.op {
            AstOp::GTExp {
                op_id: Some(id),
                base,
                ..
            } => {
                debug_assert!(
                    last_gt_exp.is_none_or(|prev| prev <= *id),
                    "non-monotone GTExp OpId encountered: prev={last_gt_exp:?}, cur={id:?}"
                );
                last_gt_exp = Some(*id);
                let inst = dory_gt_exp;
                dory_gt_exp += 1;
                gt_exp_out_instance_by_value[out_idx] = Some(inst);
                let base_idx = base.0 as usize;
                if base_idx < n && gt_exp_base_instance_by_value[base_idx].is_none() {
                    gt_exp_base_instance_by_value[base_idx] = Some(inst);
                }
            }
            AstOp::GTMul {
                op_id: Some(id), ..
            } => {
                debug_assert!(
                    last_gt_mul.is_none_or(|prev| prev <= *id),
                    "non-monotone GTMul OpId encountered: prev={last_gt_mul:?}, cur={id:?}"
                );
                last_gt_mul = Some(*id);
                let inst = dory_gt_mul;
                dory_gt_mul += 1;
                gt_mul_out_instance_by_value[out_idx] = Some(inst);
            }
            AstOp::G1ScalarMul {
                op_id: Some(id), ..
            } => {
                debug_assert!(
                    last_g1_smul.is_none_or(|prev| prev <= *id),
                    "non-monotone G1ScalarMul OpId encountered: prev={last_g1_smul:?}, cur={id:?}"
                );
                last_g1_smul = Some(*id);
                let inst = dory_g1_smul;
                dory_g1_smul += 1;
                g1_smul_out_instance_by_value[out_idx] = Some(inst);
            }
            AstOp::G2ScalarMul {
                op_id: Some(id), ..
            } => {
                debug_assert!(
                    last_g2_smul.is_none_or(|prev| prev <= *id),
                    "non-monotone G2ScalarMul OpId encountered: prev={last_g2_smul:?}, cur={id:?}"
                );
                last_g2_smul = Some(*id);
                let inst = dory_g2_smul;
                dory_g2_smul += 1;
                g2_smul_out_instance_by_value[out_idx] = Some(inst);
            }
            AstOp::G1Add {
                op_id: Some(id), ..
            } => {
                debug_assert!(
                    last_g1_add.is_none_or(|prev| prev <= *id),
                    "non-monotone G1Add OpId encountered: prev={last_g1_add:?}, cur={id:?}"
                );
                last_g1_add = Some(*id);
                let inst = dory_g1_add;
                dory_g1_add += 1;
                g1_add_out_instance_by_value[out_idx] = Some(inst);
            }
            AstOp::G2Add {
                op_id: Some(id), ..
            } => {
                debug_assert!(
                    last_g2_add.is_none_or(|prev| prev <= *id),
                    "non-monotone G2Add OpId encountered: prev={last_g2_add:?}, cur={id:?}"
                );
                last_g2_add = Some(*id);
                let inst = dory_g2_add;
                dory_g2_add += 1;
                g2_add_out_instance_by_value[out_idx] = Some(inst);
            }
            _ => {}
        }
    }

    let mut plan = WiringPlan::default();

    // --- AST internal dataflow edges (copy constraints) ---
    for node in &ast.nodes {
        match &node.op {
            AstOp::GTMul {
                op_id: Some(_),
                lhs,
                rhs,
                ..
            } => {
                let out_idx = node.out.0 as usize;
                let Some(mul_instance) = gt_mul_out_instance_by_value[out_idx] else {
                    continue;
                };
                if let Some(src) = gt_producer_from_value(
                    ast,
                    &gt_exp_out_instance_by_value,
                    &gt_mul_out_instance_by_value,
                    &gt_exp_base_instance_by_value,
                    *lhs,
                ) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtMulLhs {
                            instance: mul_instance,
                        },
                    });
                } else {
                    let src = match ast.nodes.get(lhs.0 as usize).map(|n| &n.op) {
                        Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                        Some(op) => format!("AST op {op:?}"),
                        None => "out of bounds".to_string(),
                    };
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired GTMul lhs value {lhs:?} ({src})"
                    )));
                }

                if let Some(src) = gt_producer_from_value(
                    ast,
                    &gt_exp_out_instance_by_value,
                    &gt_mul_out_instance_by_value,
                    &gt_exp_base_instance_by_value,
                    *rhs,
                ) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtMulRhs {
                            instance: mul_instance,
                        },
                    });
                } else {
                    let src = match ast.nodes.get(rhs.0 as usize).map(|n| &n.op) {
                        Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                        Some(op) => format!("AST op {op:?}"),
                        None => "out of bounds".to_string(),
                    };
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired GTMul rhs value {rhs:?} ({src})"
                    )));
                }
            }
            AstOp::G1Add {
                op_id: Some(_),
                a,
                b,
                ..
            } => {
                let out_idx = node.out.0 as usize;
                let Some(add_instance) = g1_add_out_instance_by_value[out_idx] else {
                    continue;
                };
                let a_src = g1_value_from_output(
                    ast,
                    &g1_smul_out_instance_by_value,
                    &g1_add_out_instance_by_value,
                    *a,
                )
                .or_else(|| {
                    matches!(
                        ast.nodes.get(a.0 as usize).map(|n| &n.op),
                        Some(AstOp::Input { .. })
                    )
                    .then_some(G1ValueRef::G1Input { value_id: a.0 })
                });
                if let Some(src) = a_src {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1AddInP {
                            instance: add_instance,
                        },
                    });
                } else {
                    let src = match ast.nodes.get(a.0 as usize).map(|n| &n.op) {
                        Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                        Some(op) => format!("AST op {op:?}"),
                        None => "out of bounds".to_string(),
                    };
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G1Add input a value {a:?} ({src})"
                    )));
                }
                let b_src = g1_value_from_output(
                    ast,
                    &g1_smul_out_instance_by_value,
                    &g1_add_out_instance_by_value,
                    *b,
                )
                .or_else(|| {
                    matches!(
                        ast.nodes.get(b.0 as usize).map(|n| &n.op),
                        Some(AstOp::Input { .. })
                    )
                    .then_some(G1ValueRef::G1Input { value_id: b.0 })
                });
                if let Some(src) = b_src {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1AddInQ {
                            instance: add_instance,
                        },
                    });
                } else {
                    let src = match ast.nodes.get(b.0 as usize).map(|n| &n.op) {
                        Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                        Some(op) => format!("AST op {op:?}"),
                        None => "out of bounds".to_string(),
                    };
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G1Add input b value {b:?} ({src})"
                    )));
                }
            }
            AstOp::G2Add {
                op_id: Some(_),
                a,
                b,
                ..
            } => {
                let out_idx = node.out.0 as usize;
                let Some(add_instance) = g2_add_out_instance_by_value[out_idx] else {
                    continue;
                };
                let a_src = g2_value_from_output(
                    ast,
                    &g2_smul_out_instance_by_value,
                    &g2_add_out_instance_by_value,
                    *a,
                )
                .or_else(|| {
                    matches!(
                        ast.nodes.get(a.0 as usize).map(|n| &n.op),
                        Some(AstOp::Input { .. })
                    )
                    .then_some(G2ValueRef::G2Input { value_id: a.0 })
                });
                if let Some(src) = a_src {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2AddInP {
                            instance: add_instance,
                        },
                    });
                } else {
                    let src = match ast.nodes.get(a.0 as usize).map(|n| &n.op) {
                        Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                        Some(op) => format!("AST op {op:?}"),
                        None => "out of bounds".to_string(),
                    };
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G2Add input a value {a:?} ({src})"
                    )));
                }
                let b_src = g2_value_from_output(
                    ast,
                    &g2_smul_out_instance_by_value,
                    &g2_add_out_instance_by_value,
                    *b,
                )
                .or_else(|| {
                    matches!(
                        ast.nodes.get(b.0 as usize).map(|n| &n.op),
                        Some(AstOp::Input { .. })
                    )
                    .then_some(G2ValueRef::G2Input { value_id: b.0 })
                });
                if let Some(src) = b_src {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2AddInQ {
                            instance: add_instance,
                        },
                    });
                } else {
                    let src = match ast.nodes.get(b.0 as usize).map(|n| &n.op) {
                        Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                        Some(op) => format!("AST op {op:?}"),
                        None => "out of bounds".to_string(),
                    };
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G2Add input b value {b:?} ({src})"
                    )));
                }
            }
            _ => {}
        }
    }

    // --- Base/point binding edges (bind non-input hints) ---
    for node in &ast.nodes {
        match &node.op {
            AstOp::GTExp {
                op_id: Some(_),
                base,
                ..
            } => {
                let out_idx = node.out.0 as usize;
                let Some(exp_instance) = gt_exp_out_instance_by_value[out_idx] else {
                    continue;
                };
                if let Some(src) = gt_producer_from_value(
                    ast,
                    &gt_exp_out_instance_by_value,
                    &gt_mul_out_instance_by_value,
                    &gt_exp_base_instance_by_value,
                    *base,
                ) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtExpBase {
                            instance: exp_instance,
                        },
                    });
                } else {
                    // CRITICAL: if we don't wire a GTExp base, the committed Base row can become
                    // unconstrained once Stage-2 `B-\\hat B` binding is removed.
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired GTExp base value {base:?} (GTExp instance {exp_instance})"
                    )));
                }
            }
            AstOp::G1ScalarMul {
                op_id: Some(_),
                point,
                ..
            } => {
                let out_idx = node.out.0 as usize;
                let Some(smul_instance) = g1_smul_out_instance_by_value[out_idx] else {
                    continue;
                };
                if let Some(src) = g1_value_from_output(
                    ast,
                    &g1_smul_out_instance_by_value,
                    &g1_add_out_instance_by_value,
                    *point,
                ) {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1ScalarMulBase {
                            instance: smul_instance,
                        },
                    });
                } else if is_ast_input(ast, *point) {
                    // Bind committed G1 scalar-mul base rows to the public Dory input value.
                    plan.g1.push(G1WiringEdge {
                        src: G1ValueRef::G1ScalarMulBaseBoundary {
                            instance: smul_instance,
                        },
                        dst: G1ValueRef::G1ScalarMulBase {
                            instance: smul_instance,
                        },
                    });
                } else {
                    // CRITICAL: without a base binding edge, a prover can pick an arbitrary committed
                    // base point for this scalar-mul constraint, which is then unconstrained w.r.t.
                    // the Dory AST value graph.
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G1ScalarMul base value {point:?} (G1ScalarMul instance {smul_instance})"
                    )));
                }
            }
            AstOp::G2ScalarMul {
                op_id: Some(_),
                point,
                ..
            } => {
                let out_idx = node.out.0 as usize;
                let Some(smul_instance) = g2_smul_out_instance_by_value[out_idx] else {
                    continue;
                };
                if let Some(src) = g2_value_from_output(
                    ast,
                    &g2_smul_out_instance_by_value,
                    &g2_add_out_instance_by_value,
                    *point,
                ) {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2ScalarMulBase {
                            instance: smul_instance,
                        },
                    });
                } else if is_ast_input(ast, *point) {
                    // Bind committed G2 scalar-mul base rows to the public Dory input value.
                    plan.g2.push(G2WiringEdge {
                        src: G2ValueRef::G2ScalarMulBaseBoundary {
                            instance: smul_instance,
                        },
                        dst: G2ValueRef::G2ScalarMulBase {
                            instance: smul_instance,
                        },
                    });
                } else {
                    // CRITICAL: without a base binding edge, a prover can pick an arbitrary committed
                    // base point for this scalar-mul constraint, which is then unconstrained w.r.t.
                    // the Dory AST value graph.
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G2ScalarMul base value {point:?} (G2ScalarMul instance {smul_instance})"
                    )));
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
        .find(|c| matches!(c, AstConstraint::AssertEq { .. }))
        .map(|c| match c {
            AstConstraint::AssertEq { lhs, rhs, .. } => (*lhs, *rhs),
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
        let src = g1_value_from_output(
            ast,
            &g1_smul_out_instance_by_value,
            &g1_add_out_instance_by_value,
            *vid,
        )
        .or_else(|| {
            matches!(
                ast.nodes.get(vid.0 as usize).map(|n| &n.op),
                Some(AstOp::Input { .. })
            )
            .then_some(G1ValueRef::G1Input { value_id: vid.0 })
        });
        let Some(src) = src else {
            let src = match ast.nodes.get(vid.0 as usize).map(|n| &n.op) {
                Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                Some(op) => format!("AST op {op:?}"),
                None => "out of bounds".to_string(),
            };
            return Err(ProofVerifyError::DoryError(format!(
                "unwired pairing boundary G1 input {i} value {vid:?} ({src})"
            )));
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
        let src = g2_value_from_output(
            ast,
            &g2_smul_out_instance_by_value,
            &g2_add_out_instance_by_value,
            *vid,
        )
        .or_else(|| {
            matches!(
                ast.nodes.get(vid.0 as usize).map(|n| &n.op),
                Some(AstOp::Input { .. })
            )
            .then_some(G2ValueRef::G2Input { value_id: vid.0 })
        });
        let Some(src) = src else {
            let src = match ast.nodes.get(vid.0 as usize).map(|n| &n.op) {
                Some(AstOp::Input { source }) => format!("AST input source {source:?}"),
                Some(op) => format!("AST op {op:?}"),
                None => "out of bounds".to_string(),
            };
            return Err(ProofVerifyError::DoryError(format!(
                "unwired pairing boundary G2 input {i} value {vid:?} ({src})"
            )));
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
    if let Some(src) = gt_producer_from_value(
        ast,
        &gt_exp_out_instance_by_value,
        &gt_mul_out_instance_by_value,
        &gt_exp_base_instance_by_value,
        rhs_id,
    ) {
        plan.gt.push(GtWiringEdge {
            src,
            dst: GtConsumer::PairingBoundaryRhs,
        });
    } else {
        return Err(ProofVerifyError::DoryError(format!(
            "unwired pairing RHS value {rhs_id:?}"
        )));
    }

    // --- Combine-commitments wiring (GT) + binding to joint commitment ---
    //
    // Combine constraints are appended *after* Dory-traced constraints:
    // - first: `combine_leaves` GTExp instances
    // - then:  `combine_leaves-1` GTMul instances in deterministic balanced-fold order.
    if combine_leaves > 0 {
        let combine_exp_start = dory_gt_exp;
        let combine_mul_start = dory_gt_mul;
        let expected_mul_count = CombineDag::new(combine_leaves).num_muls_total();
        // Bind each combine-leaf GTExp base to its boundary constant.
        //
        // NOTE: after Stage-2 `B-\\hat B` is removed, these bases must be explicitly anchored,
        // otherwise the committed `Base` rows for combine GTExp instances can become unconstrained.
        for i in 0..combine_leaves {
            let inst = combine_exp_start + i;
            plan.gt.push(GtWiringEdge {
                src: GtProducer::GtExpBase { instance: inst },
                dst: GtConsumer::GtExpBase { instance: inst },
            });
        }

        let mut nodes: Vec<GtProducer> = (0..combine_leaves)
            .map(|i| GtProducer::GtExpRho {
                instance: combine_exp_start + i,
            })
            .collect();
        let mut mul_idx = 0usize;
        while nodes.len() > 1 {
            let mut next: Vec<GtProducer> = Vec::with_capacity(nodes.len().div_ceil(2));
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

    // --- Pairing recursion wiring (Multi-Miller loop -> GTMul chain -> pairing Miller rhs) ---
    {
        // When pairing recursion is enabled, the constraint list includes:
        // - 3 `ConstraintType::MultiMillerLoop` instances (one per pairing pair)
        // - 2 trailing `ConstraintType::GtMul` instances to multiply the 3 Miller outputs.
        //
        // We wire:
        //   out0 -> mul0.lhs
        //   out1 -> mul0.rhs
        //   mul0.result -> mul1.lhs
        //   out2 -> mul1.rhs
        //   mul1.result -> PairingBoundaryMillerRhs
        let miller_constraint_indices: Vec<usize> = constraint_types
            .iter()
            .enumerate()
            .filter_map(|(i, ct)| matches!(ct, ConstraintType::MultiMillerLoop).then_some(i))
            .collect();
        if miller_constraint_indices.len() == 3 {
            let num_gt_mul_total = constraint_types
                .iter()
                .filter(|ct| matches!(ct, ConstraintType::GtMul))
                .count();
            if num_gt_mul_total >= 2 {
                let mul0 = num_gt_mul_total - 2;
                let mul1 = num_gt_mul_total - 1;
                let out0 = GtProducer::MultiMillerLoopOut {
                    instance: miller_constraint_indices[0],
                };
                let out1 = GtProducer::MultiMillerLoopOut {
                    instance: miller_constraint_indices[1],
                };
                let out2 = GtProducer::MultiMillerLoopOut {
                    instance: miller_constraint_indices[2],
                };

                plan.gt.push(GtWiringEdge {
                    src: out0,
                    dst: GtConsumer::GtMulLhs { instance: mul0 },
                });
                plan.gt.push(GtWiringEdge {
                    src: out1,
                    dst: GtConsumer::GtMulRhs { instance: mul0 },
                });
                plan.gt.push(GtWiringEdge {
                    src: GtProducer::GtMulResult { instance: mul0 },
                    dst: GtConsumer::GtMulLhs { instance: mul1 },
                });
                plan.gt.push(GtWiringEdge {
                    src: out2,
                    dst: GtConsumer::GtMulRhs { instance: mul1 },
                });
                plan.gt.push(GtWiringEdge {
                    src: GtProducer::GtMulResult { instance: mul1 },
                    dst: GtConsumer::PairingBoundaryMillerRhs,
                });
            }
        }
    }

    // Canonical edge ordering (must match prover and verifier): stable sort by dst then src.
    plan.gt.sort_by_key(|e| (e.dst, e.src));
    plan.g1.sort_by_key(|e| (e.dst, e.src));
    plan.g2.sort_by_key(|e| (e.dst, e.src));

    Ok(plan)
}
