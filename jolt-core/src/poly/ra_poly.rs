use rayon::prelude::*;
use std::{iter::zip, mem, sync::{Arc, OnceLock}};

use allocative::Allocative;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
};

// Shared tables for booleanity optimization: compute once, reuse across all ra_i
#[derive(Allocative, Debug, Default, PartialEq)]
pub struct RaSharedTables<F: JoltField> {
    // Base table: index x stores eq(x, r_address_chunk)
    F: Arc<Vec<F>>,

    // Lazily materialized derived tables, computed exactly once across all handles
    F_0: OnceLock<Arc<Vec<F>>>,
    F_1: OnceLock<Arc<Vec<F>>>,
    F_00: OnceLock<Arc<Vec<F>>>,
    F_01: OnceLock<Arc<Vec<F>>>,
    F_10: OnceLock<Arc<Vec<F>>>,
    F_11: OnceLock<Arc<Vec<F>>>,

    // Consistency guards
    r0: OnceLock<F::Challenge>,
    r1: OnceLock<F::Challenge>,
    order: OnceLock<BindingOrder>,
}

impl<F: JoltField> RaSharedTables<F> {
    pub fn new(eq_evals: Vec<F>) -> Self {
        Self {
            F: Arc::new(eq_evals),
            F_0: OnceLock::new(),
            F_1: OnceLock::new(),
            F_00: OnceLock::new(),
            F_01: OnceLock::new(),
            F_10: OnceLock::new(),
            F_11: OnceLock::new(),
            r0: OnceLock::new(),
            r1: OnceLock::new(),
            order: OnceLock::new(),
        }
    }

    #[inline]
    pub fn base(&self) -> &[F] { &self.F }

    pub fn ensure_round2(&self, r0: F::Challenge, order: BindingOrder) {
        if let Some(x) = self.r0.get() { debug_assert!(*x == r0); }
        if let Some(x) = self.order.get() { debug_assert!(*x == order); }
        self.r0.get_or_init(|| r0);
        self.order.get_or_init(|| order);

        let eq0 = EqPolynomial::mle(&[F::zero()], &[r0]);
        let eq1 = EqPolynomial::mle(&[F::one()], &[r0]);

        let make = |lock: &OnceLock<Arc<Vec<F>>>, scalar: F| {
            lock.get_or_init(|| {
                let mut v: Vec<F> = self.F.as_ref().clone();
                v.par_iter_mut().for_each(|x| *x *= scalar);
                Arc::new(v)
            });
        };
        make(&self.F_0, eq0);
        make(&self.F_1, eq1);
    }

    pub fn ensure_round3(&self, r1: F::Challenge) {
        if let Some(x) = self.r1.get() { debug_assert!(*x == r1); }
        self.r1.get_or_init(|| r1);

        let eq0 = EqPolynomial::mle(&[F::zero()], &[r1]);
        let eq1 = EqPolynomial::mle(&[F::one()], &[r1]);

        let F_0 = self.F_0.get().expect("ensure_round2 first").clone();
        let F_1 = self.F_1.get().expect("ensure_round2 first").clone();

        let make_from = |src: &Arc<Vec<F>>, lock: &OnceLock<Arc<Vec<F>>>, scalar: F| {
            lock.get_or_init(|| {
                let mut v: Vec<F> = src.as_ref().clone();
                v.par_iter_mut().for_each(|x| *x *= scalar);
                Arc::new(v)
            });
        };
        make_from(&F_0, &self.F_00, eq0);
        make_from(&F_0, &self.F_01, eq1);
        make_from(&F_1, &self.F_10, eq0);
        make_from(&F_1, &self.F_11, eq1);
    }

    #[inline]
    pub fn round2(&self) -> (&[F], &[F]) {
        (self.F_0.get().unwrap(), self.F_1.get().unwrap())
    }

    #[inline]
    pub fn round3(&self) -> (&[F], &[F], &[F], &[F]) {
        (
            self.F_00.get().unwrap(),
            self.F_01.get().unwrap(),
            self.F_10.get().unwrap(),
            self.F_11.get().unwrap(),
        )
    }

    #[inline]
    pub fn order(&self) -> BindingOrder {
        *self.order.get().expect("order set in ensure_round2")
    }
}

#[derive(Allocative, Clone, Debug)]
enum RaBaseTable<F: JoltField> {
    Owned(Vec<F>),
    Shared(Arc<RaSharedTables<F>>),
}

#[derive(Allocative, Clone, Debug)]
enum RaRound2Tables<F: JoltField> {
    Owned { F_0: Vec<F>, F_1: Vec<F> },
    Shared(Arc<RaSharedTables<F>>),
}

#[derive(Allocative, Clone, Debug)]
enum RaRound3Tables<F: JoltField> {
    Owned { F_00: Vec<F>, F_01: Vec<F>, F_10: Vec<F>, F_11: Vec<F> },
    Shared(Arc<RaSharedTables<F>>),
}

/// Represents the state of an `ra_i` polynomial during the last log(T) sumcheck rounds.
///
/// The first two rounds are specialized to reduce the amount of allocated memory.
#[derive(Allocative, Clone, Debug)]
pub enum RaPolynomial<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    None,
    Round1(RaPolynomialRound1<I, F>),
    Round2(RaPolynomialRound2<I, F>),
    Round3(RaPolynomialRound3<I, F>),
    RoundN(MultilinearPolynomial<F>),
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> PartialEq
    for RaPolynomial<I, F>
{
    fn eq(&self, other: &Self) -> bool {
        // Helper to compare lookup indices without requiring I: PartialEq
        let indices_equal = |a: &Arc<Vec<Option<I>>>, b: &Arc<Vec<Option<I>>>| -> bool {
            if a.len() != b.len() {
                return false;
            }
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.map(|i| i.into()) == y.map(|i| i.into()))
        };

        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Round1(a), Self::Round1(b)) => {
                let a_base: &[F] = match &a.base {
                    RaBaseTable::Owned(v) => v,
                    RaBaseTable::Shared(state) => state.base(),
                };
                let b_base: &[F] = match &b.base {
                    RaBaseTable::Owned(v) => v,
                    RaBaseTable::Shared(state) => state.base(),
                };
                a_base == b_base && indices_equal(&a.lookup_indices, &b.lookup_indices)
            }
            (Self::Round2(a), Self::Round2(b)) => {
                if a.binding_order != b.binding_order || a.r0 != b.r0 {
                    return false;
                }
                let (a0, a1): (&[F], &[F]) = match &a.tables {
                    RaRound2Tables::Owned { F_0, F_1 } => (F_0, F_1),
                    RaRound2Tables::Shared(state) => state.round2(),
                };
                let (b0, b1): (&[F], &[F]) = match &b.tables {
                    RaRound2Tables::Owned { F_0, F_1 } => (F_0, F_1),
                    RaRound2Tables::Shared(state) => state.round2(),
                };
                a0 == b0 && a1 == b1 && indices_equal(&a.lookup_indices, &b.lookup_indices)
            }
            (Self::Round3(a), Self::Round3(b)) => {
                if a.binding_order != b.binding_order || a.r1 != b.r1 {
                    return false;
                }
                let (a00, a01, a10, a11): (&[F], &[F], &[F], &[F]) = match &a.tables {
                    RaRound3Tables::Owned { F_00, F_01, F_10, F_11 } => (F_00, F_01, F_10, F_11),
                    RaRound3Tables::Shared(state) => state.round3(),
                };
                let (b00, b01, b10, b11): (&[F], &[F], &[F], &[F]) = match &b.tables {
                    RaRound3Tables::Owned { F_00, F_01, F_10, F_11 } => (F_00, F_01, F_10, F_11),
                    RaRound3Tables::Shared(state) => state.round3(),
                };
                a00 == b00
                    && a01 == b01
                    && a10 == b10
                    && a11 == b11
                    && indices_equal(&a.lookup_indices, &b.lookup_indices)
            }
            (Self::RoundN(a), Self::RoundN(b)) => a == b,
            _ => false,
        }
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> RaPolynomial<I, F> {
    /// Construct a new `RaPolynomial` from a lookup indices and a vector of eq evaluations.
    /// This owns the eq evaluations. Use `new_shared` to construct a `RaPolynomial` that
    /// shares the eq evaluations with other `RaPolynomial`s.
    pub fn new(lookup_indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::new_owned(lookup_indices, eq_evals)
    }

    pub fn new_owned(lookup_indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::Round1(RaPolynomialRound1 { base: RaBaseTable::Owned(eq_evals), lookup_indices })
    }

    pub fn new_shared(lookup_indices: Arc<Vec<Option<I>>>, shared: Arc<RaSharedTables<F>>) -> Self {
        Self::Round1(RaPolynomialRound1 { base: RaBaseTable::Shared(shared), lookup_indices })
    }

    #[inline]
    pub fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::None => panic!("RaPolynomial::get_bound_coeff called on None"),
            Self::Round1(mle) => mle.get_bound_coeff(j),
            Self::Round2(mle) => mle.get_bound_coeff(j),
            Self::Round3(mle) => mle.get_bound_coeff(j),
            Self::RoundN(mle) => mle.get_bound_coeff(j),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => panic!("RaPolynomial::len called on None"),
            Self::Round1(mle) => mle.len(),
            Self::Round2(mle) => mle.len(),
            Self::Round3(mle) => mle.len(),
            Self::RoundN(mle) => mle.len(),
        }
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> PolynomialBinding<F>
    for RaPolynomial<I, F>
{
    fn is_bound(&self) -> bool {
        !matches!(self, Self::Round1(_))
    }

    fn bind(&mut self, _r: F::Challenge, _order: BindingOrder) {
        unimplemented!()
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        let next = match mem::replace(self, Self::None) {
            Self::None => panic!("RaPolynomial::bind called on None"),
            Self::Round1(mle) => Self::Round2(mle.bind(r, order)),
            Self::Round2(mle) => Self::Round3(mle.bind(r, order)),
            Self::Round3(mle) => Self::RoundN(mle.bind(r, order)),
            Self::RoundN(mut mle) => {
                mle.bind_parallel(r, order);
                Self::RoundN(mle)
            }
        };
        *self = next;
    }

    fn final_sumcheck_claim(&self) -> F {
        match self {
            Self::RoundN(mle) => mle.final_sumcheck_claim(),
            _ => panic!(),
        }
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> PolynomialEvaluation<F>
    for RaPolynomial<I, F>
{
    fn evaluate<C>(&self, _r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!()
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        Self: Sized,
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!()
    }

    #[inline]
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![F::zero(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.get_bound_coeff(index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.get_bound_coeff(2 * index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(2 * index + 1);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
        };
        evals
    }
}

/// Represents MLE `ra_i` during the 1st round of the last log(T) sumcheck rounds.
#[derive(Allocative, Clone, Debug)]
pub struct RaPolynomialRound1<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
{
    // Base eq table: either owned per-poly or shared across polys
    base: RaBaseTable<F>,
    lookup_indices: Arc<Vec<Option<I>>>,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialRound1<I, F>
{
    fn len(&self) -> usize {
        self.lookup_indices.len()
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound1::bind")]
    fn bind(self, r0: F::Challenge, binding_order: BindingOrder) -> RaPolynomialRound2<I, F> {
        match self.base {
            RaBaseTable::Owned(Fs) => {
                // Construct per-handle tables.
                let eq_0_r0 = EqPolynomial::mle(&[F::zero()], &[r0]);
                let eq_1_r0 = EqPolynomial::mle(&[F::one()], &[r0]);
                let F_0 = Fs.iter().map(|v| eq_0_r0 * v).collect();
                let F_1 = Fs.iter().map(|v| eq_1_r0 * v).collect();
                drop_in_background_thread(Fs);
                RaPolynomialRound2 {
                    tables: RaRound2Tables::Owned { F_0, F_1 },
                    lookup_indices: self.lookup_indices,
                    r0,
                    binding_order,
                }
            }
            RaBaseTable::Shared(state) => {
                state.ensure_round2(r0, binding_order);
                RaPolynomialRound2 {
                    tables: RaRound2Tables::Shared(state),
                    lookup_indices: self.lookup_indices,
                    r0,
                    binding_order,
                }
            }
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        // Lookup ra_i(r, j).
        match &self.base {
            RaBaseTable::Owned(Fs) => self
                .lookup_indices
                .get(j)
                .expect("j out of bounds")
                .map_or(F::zero(), |i| Fs[i.into()]),
            RaBaseTable::Shared(state) => self
                .lookup_indices
                .get(j)
                .expect("j out of bounds")
                .map_or(F::zero(), |i| state.base()[i.into()]),
        }
    }
}

/// Represents `ra_i` during the 2nd of the last log(T) sumcheck rounds.
///
/// i.e. represents MLE `ra_i(r, r0, x)`
#[derive(Allocative, Clone, Debug)]
pub struct RaPolynomialRound2<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
{
    // Either owned per-poly tables, or shared across polys
    tables: RaRound2Tables<F>,
    lookup_indices: Arc<Vec<Option<I>>>,
    r0: F::Challenge,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialRound2<I, F>
{
    fn len(&self) -> usize {
        self.lookup_indices.len() / 2
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound2::bind")]
    fn bind(self, r1: F::Challenge, binding_order: BindingOrder) -> RaPolynomialRound3<I, F> {
        assert_eq!(binding_order, self.binding_order);
        match self.tables {
            RaRound2Tables::Owned { F_0, F_1 } => {
                // Construct lookup tables.
                let eq_0_r1 = EqPolynomial::mle(&[F::zero()], &[r1]);
                let eq_1_r1 = EqPolynomial::mle(&[F::one()], &[r1]);
                let mut F_00: Vec<F> = F_0.clone();
                let mut F_01: Vec<F> = F_0;
                let mut F_10: Vec<F> = F_1.clone();
                let mut F_11: Vec<F> = F_1;

                F_00.par_iter_mut().for_each(|f| *f *= eq_0_r1);
                F_01.par_iter_mut().for_each(|f| *f *= eq_1_r1);
                F_10.par_iter_mut().for_each(|f| *f *= eq_0_r1);
                F_11.par_iter_mut().for_each(|f| *f *= eq_1_r1);

                RaPolynomialRound3 {
                    tables: RaRound3Tables::Owned { F_00, F_01, F_10, F_11 },
                    lookup_indices: self.lookup_indices,
                    r1,
                    binding_order: self.binding_order,
                }
            }
            RaRound2Tables::Shared(state) => {
                state.ensure_round3(r1);
                RaPolynomialRound3 {
                    tables: RaRound3Tables::Shared(state),
                    lookup_indices: self.lookup_indices,
                    r1,
                    binding_order: self.binding_order,
                }
            }
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        let mid = self.lookup_indices.len() / 2;
        match (&self.tables, self.binding_order) {
            (RaRound2Tables::Owned { F_0, F_1 }, BindingOrder::HighToLow) => {
                let H_0 = self.lookup_indices[j].map_or(F::zero(), |i| F_0[i.into()]);
                let H_1 = self.lookup_indices[mid + j].map_or(F::zero(), |i| F_1[i.into()]);
                H_0 + H_1
            }
            (RaRound2Tables::Owned { F_0, F_1 }, BindingOrder::LowToHigh) => {
                let H_0 = self.lookup_indices[2 * j].map_or(F::zero(), |i| F_0[i.into()]);
                let H_1 = self.lookup_indices[2 * j + 1].map_or(F::zero(), |i| F_1[i.into()]);
                H_0 + H_1
            }
            (RaRound2Tables::Shared(state), BindingOrder::HighToLow) => {
                let (F_0, F_1) = state.round2();
                let H_0 = self.lookup_indices[j].map_or(F::zero(), |i| F_0[i.into()]);
                let H_1 = self.lookup_indices[mid + j].map_or(F::zero(), |i| F_1[i.into()]);
                H_0 + H_1
            }
            (RaRound2Tables::Shared(state), BindingOrder::LowToHigh) => {
                let (F_0, F_1) = state.round2();
                let H_0 = self.lookup_indices[2 * j].map_or(F::zero(), |i| F_0[i.into()]);
                let H_1 = self.lookup_indices[2 * j + 1].map_or(F::zero(), |i| F_1[i.into()]);
                H_0 + H_1
            }
        }
    }
}

/// Represents `ra_i` during the 3nd of the last log(T) sumcheck rounds.
///
/// i.e. represents MLE `ra_i(r, r0, x)`
#[derive(Allocative, Clone, Debug)]
pub struct RaPolynomialRound3<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
{
    // Either owned per-poly tables, or shared across polys
    tables: RaRound3Tables<F>,
    lookup_indices: Arc<Vec<Option<I>>>,
    r1: F::Challenge,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialRound3<I, F>
{
    fn len(&self) -> usize {
        self.lookup_indices.len() / 4
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound3::bind")]
    fn bind(self, r2: F::Challenge, _binding_order: BindingOrder) -> MultilinearPolynomial<F> {
        let lookup_indices = &self.lookup_indices;
        let n = lookup_indices.len() / 8;
        let mut res = unsafe_allocate_zero_vec(n);
        let chunk_size = 1 << 16;

        match self.tables {
            RaRound3Tables::Owned { F_00, F_01, F_10, F_11 } => {
                // Construct lookup tables (owned path matches original behavior).
                let eq_0_r2 = EqPolynomial::mle(&[F::zero()], &[r2]);
                let eq_1_r2 = EqPolynomial::mle(&[F::one()], &[r2]);

                let mut F_000: Vec<F> = F_00.clone();
                let mut F_001: Vec<F> = F_00;
                let mut F_010: Vec<F> = F_01.clone();
                let mut F_011: Vec<F> = F_01;
                let mut F_100: Vec<F> = F_10.clone();
                let mut F_101: Vec<F> = F_10;
                let mut F_110: Vec<F> = F_11.clone();
                let mut F_111: Vec<F> = F_11;

                F_000.par_iter_mut().for_each(|f| *f *= eq_0_r2);
                F_010.par_iter_mut().for_each(|f| *f *= eq_0_r2);
                F_100.par_iter_mut().for_each(|f| *f *= eq_0_r2);
                F_110.par_iter_mut().for_each(|f| *f *= eq_0_r2);
                F_001.par_iter_mut().for_each(|f| *f *= eq_1_r2);
                F_011.par_iter_mut().for_each(|f| *f *= eq_1_r2);
                F_101.par_iter_mut().for_each(|f| *f *= eq_1_r2);
                F_111.par_iter_mut().for_each(|f| *f *= eq_1_r2);

                // Eval ra_i(r, r0, r1, j) for all j in the hypercube.
                match self.binding_order {
                    BindingOrder::HighToLow => {
                        res.par_chunks_mut(chunk_size).enumerate().for_each(
                            |(chunk_index, evals_chunk)| {
                                for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                                    let H_000 = lookup_indices[j].map_or(F::zero(), |i| F_000[i.into()]);
                                    let H_001 = lookup_indices[j + n].map_or(F::zero(), |i| F_001[i.into()]);
                                    let H_010 = lookup_indices[j + n * 2].map_or(F::zero(), |i| F_010[i.into()]);
                                    let H_011 = lookup_indices[j + n * 3].map_or(F::zero(), |i| F_011[i.into()]);
                                    let H_100 = lookup_indices[j + n * 4].map_or(F::zero(), |i| F_100[i.into()]);
                                    let H_101 = lookup_indices[j + n * 5].map_or(F::zero(), |i| F_101[i.into()]);
                                    let H_110 = lookup_indices[j + n * 6].map_or(F::zero(), |i| F_110[i.into()]);
                                    let H_111 = lookup_indices[j + n * 7].map_or(F::zero(), |i| F_111[i.into()]);
                                    *eval = H_000 + H_010 + H_100 + H_110 + H_001 + H_011 + H_101 + H_111;
                                }
                            },
                        );
                    }
                    BindingOrder::LowToHigh => {
                        res.par_chunks_mut(chunk_size).enumerate().for_each(
                            |(chunk_index, evals_chunk)| {
                                for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                                    let H_000 = lookup_indices[8 * j].map_or(F::zero(), |i| F_000[i.into()]);
                                    let H_100 = lookup_indices[8 * j + 1].map_or(F::zero(), |i| F_100[i.into()]);
                                    let H_010 = lookup_indices[8 * j + 2].map_or(F::zero(), |i| F_010[i.into()]);
                                    let H_110 = lookup_indices[8 * j + 3].map_or(F::zero(), |i| F_110[i.into()]);
                                    let H_001 = lookup_indices[8 * j + 4].map_or(F::zero(), |i| F_001[i.into()]);
                                    let H_101 = lookup_indices[8 * j + 5].map_or(F::zero(), |i| F_101[i.into()]);
                                    let H_011 = lookup_indices[8 * j + 6].map_or(F::zero(), |i| F_011[i.into()]);
                                    let H_111 = lookup_indices[8 * j + 7].map_or(F::zero(), |i| F_111[i.into()]);
                                    *eval = H_000 + H_010 + H_100 + H_110 + H_001 + H_011 + H_101 + H_111;
                                }
                            },
                        );
                    }
                }

                drop_in_background_thread(self.lookup_indices);
                drop_in_background_thread(F_000);
                drop_in_background_thread(F_100);
                drop_in_background_thread(F_010);
                drop_in_background_thread(F_110);
                drop_in_background_thread(F_001);
                drop_in_background_thread(F_101);
                drop_in_background_thread(F_011);
                drop_in_background_thread(F_111);
            }
            RaRound3Tables::Shared(state) => {
                // No new arrays; compute with shared tables and inline multiply by eq(r2)
                let eq_0_r2 = EqPolynomial::mle(&[F::zero()], &[r2]);
                let eq_1_r2 = EqPolynomial::mle(&[F::one()], &[r2]);
                let (F_00, F_01, F_10, F_11) = state.round3();
                let order = state.order();

                match order {
                    BindingOrder::HighToLow => {
                        res.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_index, evals_chunk)| {
                            for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                                let s0 =
                                    self.lookup_indices[j].map_or(F::zero(), |i| F_00[i.into()]) +
                                    self.lookup_indices[j + 2 * n].map_or(F::zero(), |i| F_01[i.into()]) +
                                    self.lookup_indices[j + 4 * n].map_or(F::zero(), |i| F_10[i.into()]) +
                                    self.lookup_indices[j + 6 * n].map_or(F::zero(), |i| F_11[i.into()]);
                                let s1 =
                                    self.lookup_indices[j + 1 * n].map_or(F::zero(), |i| F_00[i.into()]) +
                                    self.lookup_indices[j + 3 * n].map_or(F::zero(), |i| F_01[i.into()]) +
                                    self.lookup_indices[j + 5 * n].map_or(F::zero(), |i| F_10[i.into()]) +
                                    self.lookup_indices[j + 7 * n].map_or(F::zero(), |i| F_11[i.into()]);
                                *eval = s0 * eq_0_r2 + s1 * eq_1_r2;
                            }
                        });
                    }
                    BindingOrder::LowToHigh => {
                        res.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_index, evals_chunk)| {
                            for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                                let base = 8 * j;
                                let s0 =
                                    self.lookup_indices[base + 0].map_or(F::zero(), |i| F_00[i.into()]) +
                                    self.lookup_indices[base + 1].map_or(F::zero(), |i| F_10[i.into()]) +
                                    self.lookup_indices[base + 2].map_or(F::zero(), |i| F_01[i.into()]) +
                                    self.lookup_indices[base + 3].map_or(F::zero(), |i| F_11[i.into()]);
                                let s1 =
                                    self.lookup_indices[base + 4].map_or(F::zero(), |i| F_00[i.into()]) +
                                    self.lookup_indices[base + 5].map_or(F::zero(), |i| F_10[i.into()]) +
                                    self.lookup_indices[base + 6].map_or(F::zero(), |i| F_01[i.into()]) +
                                    self.lookup_indices[base + 7].map_or(F::zero(), |i| F_11[i.into()]);
                                *eval = s0 * eq_0_r2 + s1 * eq_1_r2;
                            }
                        });
                    }
                }

                drop_in_background_thread(self.lookup_indices);
            }
        }

        res.into()
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        match (&self.tables, self.binding_order) {
            (RaRound3Tables::Owned { F_00, F_01, F_10, F_11 }, BindingOrder::HighToLow) => {
                let n = self.lookup_indices.len() / 4;
                let H_00 = self.lookup_indices[j].map_or(F::zero(), |i| F_00[i.into()]);
                let H_01 = self.lookup_indices[j + n].map_or(F::zero(), |i| F_01[i.into()]);
                let H_10 = self.lookup_indices[j + n * 2].map_or(F::zero(), |i| F_10[i.into()]);
                let H_11 = self.lookup_indices[j + n * 3].map_or(F::zero(), |i| F_11[i.into()]);
                H_00 + H_10 + H_01 + H_11
            }
            (RaRound3Tables::Owned { F_00, F_01, F_10, F_11 }, BindingOrder::LowToHigh) => {
                let H_00 = self.lookup_indices[4 * j].map_or(F::zero(), |i| F_00[i.into()]);
                let H_10 = self.lookup_indices[4 * j + 1].map_or(F::zero(), |i| F_10[i.into()]);
                let H_01 = self.lookup_indices[4 * j + 2].map_or(F::zero(), |i| F_01[i.into()]);
                let H_11 = self.lookup_indices[4 * j + 3].map_or(F::zero(), |i| F_11[i.into()]);
                H_00 + H_10 + H_01 + H_11
            }
            (RaRound3Tables::Shared(state), BindingOrder::HighToLow) => {
                let (F_00, F_01, F_10, F_11) = state.round3();
                let n = self.lookup_indices.len() / 4;
                let H_00 = self.lookup_indices[j].map_or(F::zero(), |i| F_00[i.into()]);
                let H_01 = self.lookup_indices[j + n].map_or(F::zero(), |i| F_01[i.into()]);
                let H_10 = self.lookup_indices[j + n * 2].map_or(F::zero(), |i| F_10[i.into()]);
                let H_11 = self.lookup_indices[j + n * 3].map_or(F::zero(), |i| F_11[i.into()]);
                H_00 + H_10 + H_01 + H_11
            }
            (RaRound3Tables::Shared(state), BindingOrder::LowToHigh) => {
                let (F_00, F_01, F_10, F_11) = state.round3();
                let H_00 = self.lookup_indices[4 * j].map_or(F::zero(), |i| F_00[i.into()]);
                let H_10 = self.lookup_indices[4 * j + 1].map_or(F::zero(), |i| F_10[i.into()]);
                let H_01 = self.lookup_indices[4 * j + 2].map_or(F::zero(), |i| F_01[i.into()]);
                let H_11 = self.lookup_indices[4 * j + 3].map_or(F::zero(), |i| F_11[i.into()]);
                H_00 + H_10 + H_01 + H_11
            }
        }
    }
}
