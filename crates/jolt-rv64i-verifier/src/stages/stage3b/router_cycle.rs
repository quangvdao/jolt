//! Concrete cycle reductions with shared bit and position opening points.

use std::collections::BTreeMap;
use std::ops::Range;

use jolt_claims::{NoChallenges, OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;

pub use crate::claims::router_cycle::{
    RouterCycleBranchInputClaims, RouterCycleBranchOutputClaims, RouterCycleCompareInputClaims,
    RouterCycleCompareOutputClaims, RouterCycleMemoryInputClaims, RouterCycleMemoryOutputClaims,
    RouterCycleShiftInputClaims, RouterCycleShiftOutputClaims, RouterCycleVariantInputClaims,
    RouterCycleVariantOutputClaims,
};
use crate::claims::router_cycle::{
    RouterCycleBranchSymbolic, RouterCycleCompareSymbolic, RouterCycleMemorySymbolic,
    RouterCycleShiftSymbolic, RouterCycleVariantSymbolic,
};
use crate::ids::{
    CommittedPolynomial, DerivedId, OpeningId, RelationId, Router, RouterCycleDerived,
    VirtualPolynomial,
};
use crate::points::{self, PointsError};
use crate::public::routes;

#[derive(Clone)]
struct CycleGeometry<F: JoltField> {
    r_1: Vec<F>,
    x: Vec<F>,
    word_slots: [F; 8],
    one_slot: F,
}

impl<F: JoltField> CycleGeometry<F> {
    fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>, router: Router) -> Result<Self, PointsError> {
        if x.len() != 17 {
            return Err(PointsError::Dimension {
                expected: 17,
                actual: x.len(),
            });
        }
        let mut word_slots = [F::zero(); 8];
        let mut one_slot = F::zero();
        match router {
            Router::Variant => {
                let slot = x.get(6..10).ok_or(PointsError::Dimension {
                    expected: 10,
                    actual: x.len(),
                })?;
                for (n, weight) in word_slots.iter_mut().enumerate() {
                    *weight = points::eq_index(slot, n)?;
                }
                let g = layout
                    .ram_ra()
                    .first()
                    .map(|chunk| usize::from(chunk.start()))
                    .ok_or(PointsError::MissingColumn { column: 64 })?;
                one_slot = points::eq_index(
                    x.get(..10).ok_or(PointsError::Dimension {
                        expected: 10,
                        actual: x.len(),
                    })?,
                    576 + layout.used_columns() - g,
                )?;
            }
            Router::Memory | Router::Branch => {
                let slot = x.get(12..13).ok_or(PointsError::Dimension {
                    expected: 13,
                    actual: x.len(),
                })?;
                for (n, weight) in word_slots.iter_mut().take(2).enumerate() {
                    *weight = points::eq_index(slot, n)?;
                }
            }
            Router::Compare => {
                let slot = x.get(12..14).ok_or(PointsError::Dimension {
                    expected: 14,
                    actual: x.len(),
                })?;
                for (n, weight) in word_slots.iter_mut().take(3).enumerate() {
                    *weight = points::eq_index(slot, n)?;
                }
                let bit = x.get(..6).ok_or(PointsError::Dimension {
                    expected: 6,
                    actual: x.len(),
                })?;
                one_slot = points::eq_index(bit, 0)? * points::eq_index(slot, 3)?;
            }
            Router::Shift => {}
        }
        Ok(Self {
            r_1,
            x,
            word_slots,
            one_slot,
        })
    }

    fn opening_point(&self, slots: Range<usize>, cycle: &[F]) -> Vec<F> {
        self.x
            .iter()
            .skip(slots.start)
            .take(slots.len())
            .chain(cycle)
            .copied()
            .collect()
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived identifiers fail closed as missing claims"
    )]
    fn term(&self, router: Router, id: &DerivedId, cycle: &[F]) -> Result<F, VerifierError> {
        match id {
            DerivedId::RouterCycle(owner, term) if *owner == router => match term {
                RouterCycleDerived::EqCycle => points::eq(&self.r_1, cycle).map_err(Self::error),
                RouterCycleDerived::WordSlot(n) => self
                    .word_slots
                    .get(*n)
                    .copied()
                    .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
                RouterCycleDerived::OneSlot => Ok(self.one_slot),
            },
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }

    fn error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "RouterCycle".to_owned(),
            reason: error.to_string(),
        }
    }
}

macro_rules! cached_expected_output {
    ($inputs:ident, $outputs:ident) => {
        fn expected_output(
            &self,
            input_points: &$inputs<Vec<F>>,
            output_values: &$outputs<F>,
            output_points: &$outputs<Vec<F>>,
            challenges: &NoChallenges<F>,
        ) -> Result<F, VerifierError> {
            let mut derived = BTreeMap::new();
            self.symbolic().output_expression::<F>().try_evaluate(
                |id| {
                    output_values
                        .resolve_output(id)
                        .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
                },
                |id| {
                    challenges
                        .resolve_challenge(id)
                        .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
                },
                |id| {
                    if let Some(value) = derived.get(id) {
                        return Ok(*value);
                    }
                    let value =
                        self.derive_output_term(id, input_points, output_points, challenges)?;
                    let _ = derived.insert(*id, value);
                    Ok(value)
                },
            )
        }
    };
}

/// Cycle reduction of the variant router, binding low-variable-first cycle coordinates after its fixed short slots.
/// `new` validates those slots; input values and their restricted points come from batch 3a.
#[derive(Clone)]
pub struct RouterCycleVariant<F: JoltField> {
    symbolic: RouterCycleVariantSymbolic,
    geometry: CycleGeometry<F>,
}

impl<F: JoltField> RouterCycleVariant<F> {
    /// Checks the seventeen low-variable-first short slots from batch 3a and returns `PointsError` for malformed geometry.
    /// The caller supplies the checked layout and verified batch-1 cycle point, whose width checked inputs establish.
    pub fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>) -> Result<Self, PointsError> {
        let symbolic = RouterCycleVariantSymbolic::new(r_1.len());
        Ok(Self {
            symbolic,
            geometry: CycleGeometry::new(layout, r_1, x, Router::Variant)?,
        })
    }
    /// The verified batch-1 cycle point, low variable first.
    pub fn r_1(&self) -> &[F] {
        &self.geometry.r_1
    }
    /// The seventeen low-variable-first short coordinates verified by batch 3a.
    pub fn x(&self) -> &[F] {
        &self.geometry.x
    }
    /// The router's restriction of the verified batch-3a short point.
    /// Returns `PointsError` when that restriction cannot be represented with the required slot geometry.
    pub fn input_points(&self) -> Result<RouterCycleVariantInputClaims<Vec<F>>, PointsError> {
        Ok(RouterCycleVariantInputClaims {
            fold: routes::restriction(Router::Variant, &self.geometry.x)?,
        })
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RouterCycleVariant<F> {
    cached_expected_output!(
        RouterCycleVariantInputClaims,
        RouterCycleVariantOutputClaims
    );
    type Symbolic = RouterCycleVariantSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RouterCycleVariantInputClaims<Vec<F>>,
    ) -> Result<RouterCycleVariantOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(CycleGeometry::<F>::error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RouterCycleVariantOutputClaims {
            rs1_value: self.geometry.opening_point(0..6, point),
            rs2_value: self.geometry.opening_point(0..6, point),
            rd_pre_value: self.geometry.opening_point(0..6, point),
            imm: self.geometry.opening_point(0..6, point),
            fall_through_pc: self.geometry.opening_point(0..6, point),
            pc_plus_imm: self.geometry.opening_point(0..6, point),
            pc: self.geometry.opening_point(0..6, point),
            next_pc: self.geometry.opening_point(0..6, point),
            variant_bits: self.geometry.opening_point(0..10, point),
            variant: self.geometry.opening_point(11..17, point),
        })
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RouterCycleVariantInputClaims<Vec<F>>,
        outputs: &RouterCycleVariantOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let cycle = outputs
            .variant
            .get(6..)
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        self.geometry.term(Router::Variant, id, cycle)
    }
}

/// Cycle reduction of the shift router, binding low-variable-first cycle coordinates after its fixed short slots.
/// `new` validates those slots; input values and their restricted points come from batch 3a.
#[derive(Clone)]
pub struct RouterCycleShift<F: JoltField> {
    symbolic: RouterCycleShiftSymbolic,
    geometry: CycleGeometry<F>,
}

impl<F: JoltField> RouterCycleShift<F> {
    /// Checks the seventeen low-variable-first short slots from batch 3a and returns `PointsError` for malformed geometry.
    /// The caller supplies the checked layout and verified batch-1 cycle point, whose width checked inputs establish.
    pub fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>) -> Result<Self, PointsError> {
        let symbolic = RouterCycleShiftSymbolic::new(r_1.len());
        Ok(Self {
            symbolic,
            geometry: CycleGeometry::new(layout, r_1, x, Router::Shift)?,
        })
    }
    /// The verified batch-1 cycle point, low variable first.
    pub fn r_1(&self) -> &[F] {
        &self.geometry.r_1
    }
    /// The seventeen low-variable-first short coordinates verified by batch 3a.
    pub fn x(&self) -> &[F] {
        &self.geometry.x
    }
    /// The router's restriction of the verified batch-3a short point.
    /// Returns `PointsError` when that restriction cannot be represented with the required slot geometry.
    pub fn input_points(&self) -> Result<RouterCycleShiftInputClaims<Vec<F>>, PointsError> {
        Ok(RouterCycleShiftInputClaims {
            fold: routes::restriction(Router::Shift, &self.geometry.x)?,
        })
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RouterCycleShift<F> {
    cached_expected_output!(RouterCycleShiftInputClaims, RouterCycleShiftOutputClaims);
    type Symbolic = RouterCycleShiftSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RouterCycleShiftInputClaims<Vec<F>>,
    ) -> Result<RouterCycleShiftOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(CycleGeometry::<F>::error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RouterCycleShiftOutputClaims {
            rs1_value: self.geometry.opening_point(0..6, point),
            shift_kind: self.geometry.opening_point(12..15, point),
            pos_ra_0: self.geometry.opening_point(6..9, point),
            pos_ra_1: self.geometry.opening_point(9..12, point),
        })
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RouterCycleShiftInputClaims<Vec<F>>,
        outputs: &RouterCycleShiftOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let cycle = outputs
            .shift_kind
            .get(3..)
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        self.geometry.term(Router::Shift, id, cycle)
    }
    fn aliased_output_openings() -> Vec<(OpeningId, OpeningId)> {
        vec![(
            OpeningId::virtual_polynomial(
                VirtualPolynomial::Rs1Value,
                RelationId::RouterCycleShift,
            ),
            OpeningId::virtual_polynomial(
                VirtualPolynomial::Rs1Value,
                RelationId::RouterCycleVariant,
            ),
        )]
    }
}

/// Cycle reduction of the memory router, binding low-variable-first cycle coordinates after its fixed short slots.
/// `new` validates those slots; input values and their restricted points come from batch 3a.
#[derive(Clone)]
pub struct RouterCycleMemory<F: JoltField> {
    symbolic: RouterCycleMemorySymbolic,
    geometry: CycleGeometry<F>,
}

impl<F: JoltField> RouterCycleMemory<F> {
    /// Checks the seventeen low-variable-first short slots from batch 3a and returns `PointsError` for malformed geometry.
    /// The caller supplies the checked layout and verified batch-1 cycle point, whose width checked inputs establish.
    pub fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>) -> Result<Self, PointsError> {
        let symbolic = RouterCycleMemorySymbolic::new(r_1.len());
        Ok(Self {
            symbolic,
            geometry: CycleGeometry::new(layout, r_1, x, Router::Memory)?,
        })
    }
    /// The verified batch-1 cycle point, low variable first.
    pub fn r_1(&self) -> &[F] {
        &self.geometry.r_1
    }
    /// The seventeen low-variable-first short coordinates verified by batch 3a.
    pub fn x(&self) -> &[F] {
        &self.geometry.x
    }
    /// The router's restriction of the verified batch-3a short point.
    /// Returns `PointsError` when that restriction cannot be represented with the required slot geometry.
    pub fn input_points(&self) -> Result<RouterCycleMemoryInputClaims<Vec<F>>, PointsError> {
        Ok(RouterCycleMemoryInputClaims {
            fold: routes::restriction(Router::Memory, &self.geometry.x)?,
        })
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RouterCycleMemory<F> {
    cached_expected_output!(RouterCycleMemoryInputClaims, RouterCycleMemoryOutputClaims);
    type Symbolic = RouterCycleMemorySymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RouterCycleMemoryInputClaims<Vec<F>>,
    ) -> Result<RouterCycleMemoryOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(CycleGeometry::<F>::error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RouterCycleMemoryOutputClaims {
            ram_read_value: self.geometry.opening_point(0..6, point),
            rs2_value: self.geometry.opening_point(0..6, point),
            access_kind: self.geometry.opening_point(13..17, point),
            pos_ra_0: self.geometry.opening_point(6..9, point),
        })
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RouterCycleMemoryInputClaims<Vec<F>>,
        outputs: &RouterCycleMemoryOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let cycle = outputs
            .access_kind
            .get(4..)
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        self.geometry.term(Router::Memory, id, cycle)
    }
    fn aliased_output_openings() -> Vec<(OpeningId, OpeningId)> {
        vec![
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Rs2Value,
                    RelationId::RouterCycleMemory,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Rs2Value,
                    RelationId::RouterCycleVariant,
                ),
            ),
            (
                OpeningId::committed(CommittedPolynomial::PosRa0, RelationId::RouterCycleMemory),
                OpeningId::committed(CommittedPolynomial::PosRa0, RelationId::RouterCycleShift),
            ),
        ]
    }
}

/// Cycle reduction of the compare router, binding low-variable-first cycle coordinates after its fixed short slots.
/// `new` validates those slots; input values and their restricted points come from batch 3a.
#[derive(Clone)]
pub struct RouterCycleCompare<F: JoltField> {
    symbolic: RouterCycleCompareSymbolic,
    geometry: CycleGeometry<F>,
}

impl<F: JoltField> RouterCycleCompare<F> {
    /// Checks the seventeen low-variable-first short slots from batch 3a and returns `PointsError` for malformed geometry.
    /// The caller supplies the checked layout and verified batch-1 cycle point, whose width checked inputs establish.
    pub fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>) -> Result<Self, PointsError> {
        let symbolic = RouterCycleCompareSymbolic::new(r_1.len());
        Ok(Self {
            symbolic,
            geometry: CycleGeometry::new(layout, r_1, x, Router::Compare)?,
        })
    }
    /// The verified batch-1 cycle point, low variable first.
    pub fn r_1(&self) -> &[F] {
        &self.geometry.r_1
    }
    /// The seventeen low-variable-first short coordinates verified by batch 3a.
    pub fn x(&self) -> &[F] {
        &self.geometry.x
    }
    /// The router's restriction of the verified batch-3a short point.
    /// Returns `PointsError` when that restriction cannot be represented with the required slot geometry.
    pub fn input_points(&self) -> Result<RouterCycleCompareInputClaims<Vec<F>>, PointsError> {
        Ok(RouterCycleCompareInputClaims {
            fold: routes::restriction(Router::Compare, &self.geometry.x)?,
        })
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RouterCycleCompare<F> {
    cached_expected_output!(
        RouterCycleCompareInputClaims,
        RouterCycleCompareOutputClaims
    );
    type Symbolic = RouterCycleCompareSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RouterCycleCompareInputClaims<Vec<F>>,
    ) -> Result<RouterCycleCompareOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(CycleGeometry::<F>::error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RouterCycleCompareOutputClaims {
            rs1_value: self.geometry.opening_point(0..6, point),
            rs2_value: self.geometry.opening_point(0..6, point),
            imm: self.geometry.opening_point(0..6, point),
            key_kind: self.geometry.opening_point(14..17, point),
            pos_ra_0: self.geometry.opening_point(6..9, point),
            pos_ra_1: self.geometry.opening_point(9..12, point),
        })
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RouterCycleCompareInputClaims<Vec<F>>,
        outputs: &RouterCycleCompareOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let cycle = outputs
            .key_kind
            .get(3..)
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        self.geometry.term(Router::Compare, id, cycle)
    }
    fn aliased_output_openings() -> Vec<(OpeningId, OpeningId)> {
        vec![
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Rs1Value,
                    RelationId::RouterCycleCompare,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Rs1Value,
                    RelationId::RouterCycleVariant,
                ),
            ),
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Rs2Value,
                    RelationId::RouterCycleCompare,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Rs2Value,
                    RelationId::RouterCycleVariant,
                ),
            ),
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Imm,
                    RelationId::RouterCycleCompare,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Imm,
                    RelationId::RouterCycleVariant,
                ),
            ),
            (
                OpeningId::committed(CommittedPolynomial::PosRa0, RelationId::RouterCycleCompare),
                OpeningId::committed(CommittedPolynomial::PosRa0, RelationId::RouterCycleShift),
            ),
            (
                OpeningId::committed(CommittedPolynomial::PosRa1, RelationId::RouterCycleCompare),
                OpeningId::committed(CommittedPolynomial::PosRa1, RelationId::RouterCycleShift),
            ),
        ]
    }
}

/// Cycle reduction of the branch router, binding low-variable-first cycle coordinates after its fixed short slots.
/// `new` validates those slots; input values and their restricted points come from batch 3a.
#[derive(Clone)]
pub struct RouterCycleBranch<F: JoltField> {
    symbolic: RouterCycleBranchSymbolic,
    geometry: CycleGeometry<F>,
}

impl<F: JoltField> RouterCycleBranch<F> {
    /// Checks the seventeen low-variable-first short slots from batch 3a and returns `PointsError` for malformed geometry.
    /// The caller supplies the checked layout and verified batch-1 cycle point, whose width checked inputs establish.
    pub fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>) -> Result<Self, PointsError> {
        let symbolic = RouterCycleBranchSymbolic::new(r_1.len());
        Ok(Self {
            symbolic,
            geometry: CycleGeometry::new(layout, r_1, x, Router::Branch)?,
        })
    }
    /// The verified batch-1 cycle point, low variable first.
    pub fn r_1(&self) -> &[F] {
        &self.geometry.r_1
    }
    /// The seventeen low-variable-first short coordinates verified by batch 3a.
    pub fn x(&self) -> &[F] {
        &self.geometry.x
    }
    /// The router's restriction of the verified batch-3a short point.
    /// Returns `PointsError` when that restriction cannot be represented with the required slot geometry.
    pub fn input_points(&self) -> Result<RouterCycleBranchInputClaims<Vec<F>>, PointsError> {
        Ok(RouterCycleBranchInputClaims {
            fold: routes::restriction(Router::Branch, &self.geometry.x)?,
        })
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RouterCycleBranch<F> {
    cached_expected_output!(RouterCycleBranchInputClaims, RouterCycleBranchOutputClaims);
    type Symbolic = RouterCycleBranchSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RouterCycleBranchInputClaims<Vec<F>>,
    ) -> Result<RouterCycleBranchOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(CycleGeometry::<F>::error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RouterCycleBranchOutputClaims {
            fall_through_pc: self.geometry.opening_point(0..6, point),
            pc_plus_imm: self.geometry.opening_point(0..6, point),
            branch: self.geometry.opening_point(0..0, point),
            should_branch: self.geometry.opening_point(0..0, point),
        })
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RouterCycleBranchInputClaims<Vec<F>>,
        outputs: &RouterCycleBranchOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let cycle = outputs
            .branch
            .get(0..)
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        self.geometry.term(Router::Branch, id, cycle)
    }
    fn aliased_output_openings() -> Vec<(OpeningId, OpeningId)> {
        vec![
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::FallThroughPC,
                    RelationId::RouterCycleBranch,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::FallThroughPC,
                    RelationId::RouterCycleVariant,
                ),
            ),
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::PCPlusImm,
                    RelationId::RouterCycleBranch,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::PCPlusImm,
                    RelationId::RouterCycleVariant,
                ),
            ),
        ]
    }
}
