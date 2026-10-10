//! Concrete short router relation over the shared 17 slots.
use crate::claims::router_short::RouterShortSymbolic;
pub use crate::claims::router_short::{RouterShortInputClaims, RouterShortOutputClaims};
use crate::ids::RouterShortDerived;
use crate::ids::{DerivedId, Router};
use crate::points::PointsError;
use crate::public::routes::{equality_table, restriction, short_point, RouteTensors};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_verifier::{stages::relations::ConcreteSumcheck, VerifierError};
use std::sync::Arc;

/// Reduction over the seventeen low-variable-first router slots at batch 2's column and batch 1's cycle points.
/// Its shared route tensors must be generated for the checked layout.
#[derive(Clone)]
pub struct RouterShort<F: JoltField> {
    symbolic: RouterShortSymbolic,
    w: Vec<F>,
    r_1: Vec<F>,
    routes: Arc<RouteTensors>,
    columns: Vec<F>,
}
impl<F: JoltField> RouterShort<F> {
    /// Checks that the verified batch-2 column point has ten coordinates, returning `PointsError` otherwise.
    /// The caller supplies the verified batch-1 cycle point; checked inputs establish its trace width.
    pub fn new(w: Vec<F>, r_1: Vec<F>, routes: Arc<RouteTensors>) -> Result<Self, PointsError> {
        if w.len() != 10 {
            return Err(PointsError::Dimension {
                expected: 10,
                actual: w.len(),
            });
        }
        let columns = equality_table(&w)?;
        Ok(Self {
            symbolic: RouterShortSymbolic::new(()),
            w,
            r_1,
            routes,
            columns,
        })
    }
    /// The ten low-variable-first witness-column coordinates read from batch 2.
    pub fn w(&self) -> &[F] {
        &self.w
    }
    /// The low-variable-first batch-1 cycle point retained through batch 2.
    pub fn r_1(&self) -> &[F] {
        &self.r_1
    }
    /// The checked layout's route tensors shared by all five router reductions.
    pub fn routes(&self) -> &RouteTensors {
        &self.routes
    }
    /// The consumed routed-witness point in column-then-cycle order, low variable first.
    pub fn input_points(&self) -> RouterShortInputClaims<Vec<F>> {
        RouterShortInputClaims {
            witness_routed: self.w.iter().chain(&self.r_1).copied().collect(),
        }
    }
    fn error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "RouterShort".to_owned(),
            reason: error.to_string(),
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RouterShort<F> {
    type Symbolic = RouterShortSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _: &RouterShortInputClaims<Vec<F>>,
    ) -> Result<RouterShortOutputClaims<Vec<F>>, VerifierError> {
        Ok(RouterShortOutputClaims {
            variant: restriction(Router::Variant, point).map_err(Self::error)?,
            shift: restriction(Router::Shift, point).map_err(Self::error)?,
            memory: restriction(Router::Memory, point).map_err(Self::error)?,
            compare: restriction(Router::Compare, point).map_err(Self::error)?,
            branch: restriction(Router::Branch, point).map_err(Self::error)?,
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _: &RouterShortInputClaims<Vec<F>>,
        outputs: &RouterShortOutputClaims<Vec<F>>,
        _: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::RouterShort(RouterShortDerived::RouteWeight(r)) => {
                let point = short_point(&outputs.compare).map_err(Self::error)?;
                self.routes
                    .weight(*r, &self.columns, &point)
                    .map_err(Self::error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
}
