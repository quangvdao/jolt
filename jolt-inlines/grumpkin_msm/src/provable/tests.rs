use super::{curve, fq, grumpkin_msm_2048};
use crate::types::{AffinePoint, FqLimbs, FrLimbs, JacobianPoint, LIMBS_64};

use alloc::vec;
use alloc::vec::Vec;
use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, PrimeField, UniformRand, Zero};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn fq_limbs_from_ark(f: &ark_grumpkin::Fq) -> FqLimbs {
    FqLimbs(f.into_bigint().0)
}

fn fr_limbs_from_ark(f: &ark_grumpkin::Fr) -> FrLimbs {
    FrLimbs(f.into_bigint().0)
}

fn affine_from_ark(p: &ark_grumpkin::Affine) -> AffinePoint {
    if p.is_zero() {
        AffinePoint::infinity()
    } else {
        AffinePoint {
            x: fq_limbs_from_ark(&p.x),
            y: fq_limbs_from_ark(&p.y),
            infinity: 0,
        }
    }
}

fn jac_from_ark(p: &ark_grumpkin::Projective) -> JacobianPoint {
    JacobianPoint {
        x: fq_limbs_from_ark(&p.x),
        y: fq_limbs_from_ark(&p.y),
        z: fq_limbs_from_ark(&p.z),
    }
}

fn jac_to_ark(j: &JacobianPoint) -> ark_grumpkin::Projective {
    if j.is_infinity() {
        return ark_grumpkin::Projective::zero();
    }
    let x = <ark_grumpkin::Fq as PrimeField>::from_bigint(BigInt::new(j.x.0))
        .expect("canonical fq limbs");
    let y = <ark_grumpkin::Fq as PrimeField>::from_bigint(BigInt::new(j.y.0))
        .expect("canonical fq limbs");
    let z = <ark_grumpkin::Fq as PrimeField>::from_bigint(BigInt::new(j.z.0))
        .expect("canonical fq limbs");
    ark_grumpkin::Projective { x, y, z }
}

#[test]
fn fq_ops_match_arkworks_random() {
    let mut rng = StdRng::seed_from_u64(0xF00D_F00D);
    for _ in 0..200 {
        let a = ark_grumpkin::Fq::rand(&mut rng);
        let b = ark_grumpkin::Fq::rand(&mut rng);

        let a_l = fq_limbs_from_ark(&a);
        let b_l = fq_limbs_from_ark(&b);

        let a_m = fq::FqMont::from_canonical(a_l);
        let b_m = fq::FqMont::from_canonical(b_l);

        let add = a_m.add(b_m).to_canonical();
        let sub = a_m.sub(b_m).to_canonical();
        let mul = a_m.mul(b_m).to_canonical();
        let sqr = a_m.square().to_canonical();

        let add_ref = a + b;
        let sub_ref = a - b;
        let mul_ref = a * b;
        let sqr_ref = a * a;

        assert_eq!(add, fq_limbs_from_ark(&add_ref));
        assert_eq!(sub, fq_limbs_from_ark(&sub_ref));
        assert_eq!(mul, fq_limbs_from_ark(&mul_ref));
        assert_eq!(sqr, fq_limbs_from_ark(&sqr_ref));
    }
}

#[test]
fn curve_double_matches_arkworks_random() {
    let mut rng = StdRng::seed_from_u64(0xD0_0B_1E);
    for _ in 0..50 {
        let p = ark_grumpkin::Projective::rand(&mut rng);
        let p_j = jac_from_ark(&p);

        let p_m = curve::JacobianMont {
            x: fq::FqMont::from_canonical(p_j.x),
            y: fq::FqMont::from_canonical(p_j.y),
            z: fq::FqMont::from_canonical(p_j.z),
        };
        let out = curve::double_jac(p_m).to_canonical();

        let mut expected = p;
        expected.double_in_place();
        assert_eq!(jac_to_ark(&out).into_affine(), expected.into_affine());
    }
}

#[test]
fn curve_add_mixed_matches_arkworks_random() {
    let mut rng = StdRng::seed_from_u64(0xADD1_CEED);
    for _ in 0..50 {
        let p = ark_grumpkin::Projective::rand(&mut rng);
        let q = ark_grumpkin::Projective::rand(&mut rng).into_affine();

        let p_j = jac_from_ark(&p);
        let q_a = affine_from_ark(&q);

        let p_m = curve::JacobianMont {
            x: fq::FqMont::from_canonical(p_j.x),
            y: fq::FqMont::from_canonical(p_j.y),
            z: fq::FqMont::from_canonical(p_j.z),
        };
        let q_m = curve::AffineMont::from_affine_point(&q_a);

        let out = curve::add_mixed(p_m, &q_m).to_canonical();

        let expected = p + ark_grumpkin::Projective::from(q);
        assert_eq!(jac_to_ark(&out).into_affine(), expected.into_affine());
    }
}

#[test]
fn msm_2048_matches_arkworks_once() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    let mut bases_affine = Vec::with_capacity(crate::MSM_N);
    let mut bases_limbs = Vec::with_capacity(crate::MSM_N);
    let mut scalars = Vec::with_capacity(crate::MSM_N);
    let mut scalars_limbs = Vec::with_capacity(crate::MSM_N);

    for _ in 0..crate::MSM_N {
        let p = ark_grumpkin::Projective::rand(&mut rng).into_affine();
        let s = ark_grumpkin::Fr::rand(&mut rng);
        bases_affine.push(p);
        bases_limbs.push(affine_from_ark(&p));
        scalars.push(s);
        scalars_limbs.push(fr_limbs_from_ark(&s));
    }

    let expected = ark_grumpkin::Projective::msm_unchecked(&bases_affine, &scalars);
    let got = grumpkin_msm_2048(&bases_limbs, &scalars_limbs);

    assert_eq!(jac_to_ark(&got).into_affine(), expected.into_affine());
}

#[test]
fn msm_2048_all_zero_scalars_is_infinity() {
    let mut rng = StdRng::seed_from_u64(0xBADC0DE);
    let mut bases = Vec::with_capacity(crate::MSM_N);
    let scalars = vec![FrLimbs([0u64; LIMBS_64]); crate::MSM_N];
    for _ in 0..crate::MSM_N {
        let p = ark_grumpkin::Projective::rand(&mut rng).into_affine();
        bases.push(affine_from_ark(&p));
    }

    let got = grumpkin_msm_2048(&bases, &scalars);
    assert!(got.is_infinity());
}
