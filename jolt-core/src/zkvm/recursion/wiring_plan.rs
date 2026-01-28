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

use super::CombineDag;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::proof_serialization::PairingBoundary;
use crate::zkvm::{guest_serde::GuestDeserialize, guest_serde::GuestSerialize};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use dory::recursion::ast::{AstConstraint, AstGraph, AstOp, ValueId};
use dory::recursion::OpId;
use std::io::{Read, Write};

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
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.gt.guest_serialize(w)?;
        self.g1.guest_serialize(w)?;
        self.g2.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for WiringPlan {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.src.guest_serialize(w)?;
        self.dst.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for GtWiringEdge {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for GtProducer {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            GtProducer::GtExpRho { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "GtProducer instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            GtProducer::GtMulResult { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "GtProducer instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            GtProducer::GtExpBase { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "GtProducer instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
        }
    }
}

impl GuestDeserialize for GtProducer {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
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
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        match self {
            GtConsumer::GtMulLhs { .. }
            | GtConsumer::GtMulRhs { .. }
            | GtConsumer::GtExpBase { .. } => 1 + 4,
            GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => 1,
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
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for GtConsumer {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            GtConsumer::GtMulLhs { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "GtConsumer instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            GtConsumer::GtMulRhs { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "GtConsumer instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            GtConsumer::GtExpBase { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "GtConsumer instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            GtConsumer::JointCommitment => 3u8.guest_serialize(w),
            GtConsumer::PairingBoundaryRhs => 4u8.guest_serialize(w),
        }
    }
}

impl GuestDeserialize for GtConsumer {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
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
    /// The constant value is taken from `ConstraintType::G1ScalarMul { base_point }` (local index).
    G1ScalarMulBase {
        instance: usize,
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
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.src.guest_serialize(w)?;
        self.dst.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for G1WiringEdge {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        match self {
            G1ValueRef::G1ScalarMulOut { .. }
            | G1ValueRef::G1AddOut { .. }
            | G1ValueRef::G1AddInP { .. }
            | G1ValueRef::G1AddInQ { .. }
            | G1ValueRef::G1ScalarMulBase { .. } => 1 + 4,
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
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for G1ValueRef {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            G1ValueRef::G1ScalarMulOut { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G1ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1AddOut { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G1ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1AddInP { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G1ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1AddInQ { instance } => {
                3u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G1ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::G1ScalarMulBase { instance } => {
                4u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G1ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G1ValueRef::PairingBoundaryP1 => 5u8.guest_serialize(w),
            G1ValueRef::PairingBoundaryP2 => 6u8.guest_serialize(w),
            G1ValueRef::PairingBoundaryP3 => 7u8.guest_serialize(w),
        }
    }
}

impl GuestDeserialize for G1ValueRef {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
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
    /// The constant value is taken from `ConstraintType::G2ScalarMul { base_point }` (local index).
    G2ScalarMulBase {
        instance: usize,
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
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.src.guest_serialize(w)?;
        self.dst.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for G2WiringEdge {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
            G2ValueRef::PairingBoundaryP1 => 5u8.serialize_with_mode(&mut writer, compress),
            G2ValueRef::PairingBoundaryP2 => 6u8.serialize_with_mode(&mut writer, compress),
            G2ValueRef::PairingBoundaryP3 => 7u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let _ = compress;
        match self {
            G2ValueRef::G2ScalarMulOut { .. }
            | G2ValueRef::G2AddOut { .. }
            | G2ValueRef::G2AddInP { .. }
            | G2ValueRef::G2AddInQ { .. }
            | G2ValueRef::G2ScalarMulBase { .. } => 1 + 4,
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
            5 => Self::PairingBoundaryP1,
            6 => Self::PairingBoundaryP2,
            7 => Self::PairingBoundaryP3,
            _ => return Err(SerializationError::InvalidData),
        })
    }
}

impl GuestSerialize for G2ValueRef {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            G2ValueRef::G2ScalarMulOut { instance } => {
                0u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G2ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2AddOut { instance } => {
                1u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G2ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2AddInP { instance } => {
                2u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G2ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2AddInQ { instance } => {
                3u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G2ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::G2ScalarMulBase { instance } => {
                4u8.guest_serialize(w)?;
                (u32::try_from(*instance).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "G2ValueRef instance overflow",
                    )
                })?)
                .guest_serialize(w)
            }
            G2ValueRef::PairingBoundaryP1 => 5u8.guest_serialize(w),
            G2ValueRef::PairingBoundaryP2 => 6u8.guest_serialize(w),
            G2ValueRef::PairingBoundaryP3 => 7u8.guest_serialize(w),
        }
    }
}

impl GuestDeserialize for G2ValueRef {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
            5 => Self::PairingBoundaryP1,
            6 => Self::PairingBoundaryP2,
            7 => Self::PairingBoundaryP3,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid G2ValueRef tag",
                ))
            }
        })
    }
}

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
            AstOp::GTExp {
                op_id: Some(id), ..
            } => out.gt_exp.push(*id),
            AstOp::GTMul {
                op_id: Some(id), ..
            } => out.gt_mul.push(*id),
            AstOp::G1ScalarMul {
                op_id: Some(id), ..
            } => out.g1_smul.push(*id),
            AstOp::G2ScalarMul {
                op_id: Some(id), ..
            } => out.g2_smul.push(*id),
            AstOp::G1Add {
                op_id: Some(id), ..
            } => out.g1_add.push(*id),
            AstOp::G2Add {
                op_id: Some(id), ..
            } => out.g2_add.push(*id),
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
    op_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect()
}

fn gt_producer_from_value(
    ast: &AstGraph<dory::backends::arkworks::BN254>,
    gt_exp_index: &std::collections::BTreeMap<OpId, usize>,
    gt_mul_index: &std::collections::BTreeMap<OpId, usize>,
    gt_exp_base_by_value: &std::collections::BTreeMap<ValueId, usize>,
    value: ValueId,
) -> Option<GtProducer> {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return None;
    }
    match &ast.nodes[idx].op {
        AstOp::GTExp {
            op_id: Some(id), ..
        } => gt_exp_index
            .get(id)
            .copied()
            .map(|instance| GtProducer::GtExpRho { instance }),
        AstOp::GTMul {
            op_id: Some(id), ..
        } => gt_mul_index
            .get(id)
            .copied()
            .map(|instance| GtProducer::GtMulResult { instance }),
        AstOp::Input { .. } => gt_exp_base_by_value
            .get(&value)
            .copied()
            .map(|instance| GtProducer::GtExpBase { instance }),
        _ => None,
    }
}

fn is_ast_input(ast: &AstGraph<dory::backends::arkworks::BN254>, value: ValueId) -> bool {
    let idx = value.0 as usize;
    if idx >= ast.nodes.len() {
        return false;
    }
    matches!(&ast.nodes[idx].op, AstOp::Input { .. })
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
        AstOp::G1ScalarMul {
            op_id: Some(id), ..
        } => g1_smul_index
            .get(id)
            .copied()
            .map(|instance| G1ValueRef::G1ScalarMulOut { instance }),
        AstOp::G1Add {
            op_id: Some(id), ..
        } => g1_add_index
            .get(id)
            .copied()
            .map(|instance| G1ValueRef::G1AddOut { instance }),
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
        AstOp::G2ScalarMul {
            op_id: Some(id), ..
        } => g2_smul_index
            .get(id)
            .copied()
            .map(|instance| G2ValueRef::G2ScalarMulOut { instance }),
        AstOp::G2Add {
            op_id: Some(id), ..
        } => g2_add_index
            .get(id)
            .copied()
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
    // Map ValueId(Input GT value) -> GTExp instance index whose base is that ValueId.
    // This lets us treat GT inputs as the corresponding GTExp base constant in wiring.
    let mut gt_exp_base_by_value: std::collections::BTreeMap<ValueId, usize> =
        std::collections::BTreeMap::new();
    for node in &ast.nodes {
        if let AstOp::GTExp {
            op_id: Some(id),
            base,
            ..
        } = &node.op
        {
            if let Some(&instance) = gt_exp_index.get(id) {
                gt_exp_base_by_value.entry(*base).or_insert(instance);
            }
        }
    }

    // --- AST internal dataflow edges (copy constraints) ---
    for node in &ast.nodes {
        match &node.op {
            AstOp::GTMul {
                op_id: Some(id),
                lhs,
                rhs,
                ..
            } => {
                let Some(&mul_instance) = gt_mul_index.get(id) else {
                    continue;
                };
                if let Some(src) = gt_producer_from_value(
                    ast,
                    &gt_exp_index,
                    &gt_mul_index,
                    &gt_exp_base_by_value,
                    *lhs,
                ) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtMulLhs {
                            instance: mul_instance,
                        },
                    });
                } else if !is_ast_input(ast, *lhs) {
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired GTMul lhs value {lhs:?}"
                    )));
                }

                if let Some(src) = gt_producer_from_value(
                    ast,
                    &gt_exp_index,
                    &gt_mul_index,
                    &gt_exp_base_by_value,
                    *rhs,
                ) {
                    plan.gt.push(GtWiringEdge {
                        src,
                        dst: GtConsumer::GtMulRhs {
                            instance: mul_instance,
                        },
                    });
                } else if !is_ast_input(ast, *rhs) {
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired GTMul rhs value {rhs:?}"
                    )));
                }
            }
            AstOp::G1Add {
                op_id: Some(id),
                a,
                b,
                ..
            } => {
                let Some(&add_instance) = g1_add_index.get(id) else {
                    continue;
                };
                if let Some(src) = g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *a) {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1AddInP {
                            instance: add_instance,
                        },
                    });
                } else if !is_ast_input(ast, *a) {
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G1Add input a value {a:?}"
                    )));
                }
                if let Some(src) = g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *b) {
                    plan.g1.push(G1WiringEdge {
                        src,
                        dst: G1ValueRef::G1AddInQ {
                            instance: add_instance,
                        },
                    });
                } else if !is_ast_input(ast, *b) {
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G1Add input b value {b:?}"
                    )));
                }
            }
            AstOp::G2Add {
                op_id: Some(id),
                a,
                b,
                ..
            } => {
                let Some(&add_instance) = g2_add_index.get(id) else {
                    continue;
                };
                if let Some(src) = g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *a) {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2AddInP {
                            instance: add_instance,
                        },
                    });
                } else if !is_ast_input(ast, *a) {
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G2Add input a value {a:?}"
                    )));
                }
                if let Some(src) = g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *b) {
                    plan.g2.push(G2WiringEdge {
                        src,
                        dst: G2ValueRef::G2AddInQ {
                            instance: add_instance,
                        },
                    });
                } else if !is_ast_input(ast, *b) {
                    return Err(ProofVerifyError::DoryError(format!(
                        "unwired G2Add input b value {b:?}"
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
                op_id: Some(id),
                base,
                ..
            } => {
                let Some(&exp_instance) = gt_exp_index.get(id) else {
                    continue;
                };
                if let Some(src) = gt_producer_from_value(
                    ast,
                    &gt_exp_index,
                    &gt_mul_index,
                    &gt_exp_base_by_value,
                    *base,
                ) {
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
                let Some(&smul_instance) = g1_smul_index.get(id) else {
                    continue;
                };
                if let Some(src) = g1_value_from_output(ast, &g1_smul_index, &g1_add_index, *point)
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
                let Some(&smul_instance) = g2_smul_index.get(id) else {
                    continue;
                };
                if let Some(src) = g2_value_from_output(ast, &g2_smul_index, &g2_add_index, *point)
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
    if let Some(src) = gt_producer_from_value(
        ast,
        &gt_exp_index,
        &gt_mul_index,
        &gt_exp_base_by_value,
        rhs_id,
    ) {
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

    // Canonical edge ordering (must match prover and verifier): stable sort by dst then src.
    plan.gt.sort_by_key(|e| (e.dst, e.src));
    plan.g1.sort_by_key(|e| (e.dst, e.src));
    plan.g2.sort_by_key(|e| (e.dst, e.src));

    Ok(plan)
}
