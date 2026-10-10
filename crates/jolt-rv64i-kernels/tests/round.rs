#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "test fixture failures are assertion failures"
)]

use jolt_field::F128;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};
use jolt_rv64i_kernels::oracle::mle_at;
use jolt_rv64i_kernels::round::eq::{eq_table, split_eq};
use jolt_rv64i_kernels::round::{
    coefficients_from_nodes, eval_at_node, linear_at_nodes, quadratic, quadratic_at_nodes,
    RoundError,
};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

#[test]
fn variable_order_matches_literal_vertices_and_shared_conventions() {
    let point = [ZERO, ONE];
    let table = [ZERO, ONE, ZERO, ONE];
    assert_eq!(mle_at(&table, &point).unwrap(), ZERO);
    let weights = eq_table(&point, None);
    assert_eq!(weights, [ZERO, ZERO, ONE, ZERO]);
    assert_eq!(
        weights.iter().zip(table).map(|(&w, g)| w * g).sum::<F128>(),
        ZERO
    );
    assert_eq!(
        split_eq(&point, None).unwrap().current_linear_evals(),
        (ONE, ZERO)
    );
    let shared = EqPolynomial::<F128>::evals(&point, None);
    assert_eq!(shared, [ZERO, ONE, ZERO, ZERO]);
    assert_eq!(
        shared.iter().zip(table).map(|(&w, g)| w * g).sum::<F128>(),
        ONE
    );
    assert_eq!(
        GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh).current_linear_evals(),
        (ZERO, ONE)
    );
    use jolt_rv64i_kernels::reduction::ReductionCore;
    use jolt_sumcheck::ProveRounds;
    let mut core =
        ReductionCore::new(vec![table.to_vec()], vec![(0, point.to_vec(), ONE, ZERO)]).unwrap();
    assert_eq!(
        core.prove_round(None, 0, ZERO).unwrap().coefficients(),
        [ZERO, ONE, ONE]
    );
}

#[test]
fn equality_tables_and_split_rounds_match_oracle_on_distinct_coordinates() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x8cb9);
    for variables in 1..=8 {
        let point: Vec<_> = (0..variables)
            .map(|i| F128::from_raw((u128::from(rng.next_u64()) << 64) | (i as u128 + 2)))
            .collect();
        let challenges: Vec<_> = (0..variables)
            .map(|i| F128::from_raw((u128::from(rng.next_u64()) << 64) | (i as u128 + 17)))
            .collect();
        let expected: Vec<_> = (0..1 << variables)
            .map(|index| {
                let mut vertex = vec![ZERO; 1 << variables];
                vertex[index] = ONE;
                mle_at(&vertex, &point).unwrap()
            })
            .collect();
        for scale in [None, Some(ZERO), Some(F128::from_raw(0x9753))] {
            let scale_value = scale.unwrap_or(ONE);
            let scaled: Vec<_> = expected.iter().map(|&e| scale_value * e).collect();
            assert_eq!(eq_table(&point, scale), scaled);
            let mut split = split_eq(&point, scale).unwrap();
            for round in 0..variables {
                let (at_zero, at_one) = split.current_linear_evals();
                for endpoint in [0, 1] {
                    let mut sum = ZERO;
                    for suffix in 0..1 << (variables - round - 1) {
                        let mut at = challenges[..round].to_vec();
                        at.push(F128::from_raw(endpoint as u128));
                        at.extend(
                            (0..variables - round - 1)
                                .map(|i| F128::from_raw(((suffix >> i) & 1) as u128)),
                        );
                        let value = mle_at(&scaled, &at).unwrap();
                        let inner = split.e_in_current();
                        let outer = split.e_out_current();
                        let linear = if endpoint == 0 { at_zero } else { at_one };
                        assert_eq!(
                            value,
                            linear * inner[suffix % inner.len()] * outer[suffix / inner.len()]
                        );
                        sum += value;
                    }
                    assert_eq!(sum, if endpoint == 0 { at_zero } else { at_one });
                }
                split.bind(challenges[round]);
            }
            assert_eq!(
                split.current_scalar(),
                mle_at(&scaled, &challenges).unwrap()
            );
        }
    }
    for scale in [None, Some(ZERO), Some(F128::from_raw(19))] {
        assert_eq!(eq_table(&[], scale), [scale.unwrap_or(ONE)]);
    }
}

#[test]
fn seeded_nodes_recover_coefficients_for_every_degree() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7743);
    for degree in 2..=8 {
        for _ in 0..32 {
            let coefficients: Vec<_> = (0..=degree)
                .map(|_| {
                    F128::from_raw((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64()))
                })
                .collect();
            let at = |raw| {
                coefficients
                    .iter()
                    .rev()
                    .fold(ZERO, |acc, &c| acc * F128::from_raw(raw) + c)
            };
            let nodes: Vec<_> = (2..degree).map(|raw| at(raw as u128)).collect();
            let recovered = coefficients_from_nodes(
                degree,
                coefficients[0],
                coefficients[degree],
                at(1),
                &nodes,
            )
            .unwrap();
            assert_eq!(recovered.len(), degree + 1);
            assert_eq!(recovered, coefficients);
            for node in 0..=8 {
                assert_eq!(eval_at_node(&coefficients, node), at(u128::from(node)));
            }
        }
    }
}

#[test]
fn quadratic_coefficients_and_nodes_equal_coefficient_products() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x39ed);
    for _ in 0..128 {
        let left = [
            F128::from_raw(u128::from(rng.next_u64()) << 64),
            F128::from_raw(u128::from(rng.next_u64())),
        ];
        let right = [
            F128::from_raw(u128::from(rng.next_u64())),
            F128::from_raw(u128::from(rng.next_u64()) << 64),
        ];
        let product = quadratic(left, right);
        assert_eq!(
            product,
            [
                left[0] * right[0],
                left[0] * right[1] + left[1] * right[0],
                left[1] * right[1]
            ]
        );
        for node in 0..=8 {
            let x = F128::from_raw(u128::from(node));
            assert_eq!(
                eval_at_node(&product, node),
                (left[0] + x * left[1]) * (right[0] + x * right[1])
            );
        }
    }
}

#[test]
fn shared_node_values_equal_multiplication_evaluation() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x621c);
    for _ in 0..128 {
        let [a, b, c] = std::array::from_fn(|_| {
            F128::from_raw((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64()))
        });
        for coefficients in [[a, b, c], [a, b, ZERO], [a, ZERO, c], [a, a, a]] {
            for (i, value) in quadratic_at_nodes(coefficients).into_iter().enumerate() {
                let x = F128::from_raw((i + 2) as u128);
                assert_eq!(
                    value,
                    coefficients[0] + coefficients[1] * x + coefficients[2] * x * x
                );
            }
            for (i, value) in linear_at_nodes([coefficients[0], coefficients[1]])
                .into_iter()
                .enumerate()
            {
                let x = F128::from_raw((i + 2) as u128);
                assert_eq!(value, coefficients[0] + coefficients[1] * x);
            }
        }
    }
}

#[test]
fn invalid_round_geometry_returns_typed_errors() {
    for degree in [1, 9, usize::MAX] {
        assert_eq!(
            coefficients_from_nodes(degree, ZERO, ZERO, ZERO, &[]),
            Err(RoundError::Degree { degree })
        );
    }
    assert_eq!(
        coefficients_from_nodes(5, ZERO, ZERO, ZERO, &[ZERO; 2]),
        Err(RoundError::Nodes {
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(
        coefficients_from_nodes(2, ZERO, ZERO, ZERO, &[ZERO]),
        Err(RoundError::Nodes {
            expected: 0,
            actual: 1
        })
    );
    assert!(matches!(split_eq(&[], None), Err(RoundError::EmptyPoint)));
}
