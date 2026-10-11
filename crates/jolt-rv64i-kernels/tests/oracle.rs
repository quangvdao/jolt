#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "oracle fixtures have checked dimensions"
)]

use jolt_field::F128;
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial, OracleError};

#[test]
fn multilinear_extension_interpolates_boolean_vertices() {
    for variables in 0..=8 {
        let table: Vec<_> = (0..1 << variables)
            .map(|i| F128::from_raw((i as u128) << 73 | 0x953))
            .collect();
        for (index, &value) in table.iter().enumerate() {
            let point: Vec<_> = (0..variables)
                .map(|bit| F128::from_raw(((index >> bit) & 1) as u128))
                .collect();
            assert_eq!(mle_at(&table, &point).unwrap(), value);
        }
    }
}

#[test]
fn malformed_oracle_inputs_return_typed_errors() {
    let z = F128::from_raw(0);
    assert!(matches!(
        mle_at(&[], &[]),
        Err(OracleError::TableLength { length: 0, .. })
    ));
    assert_eq!(
        mle_at(&[z; 2], &[]),
        Err(OracleError::PointLength {
            expected: 1,
            actual: 0
        })
    );
    assert!(matches!(
        round_polynomial(&[], &[], 2, |_| z),
        Err(OracleError::EmptyLeaves)
    ));
    assert!(matches!(
        round_polynomial(&[&[z; 2], &[z; 4]], &[], 2, |_| z),
        Err(OracleError::LeafLength { leaf: 1, .. })
    ));
    assert!(matches!(
        round_polynomial(&[&[z; 2]], &[z], 2, |_| z),
        Err(OracleError::NoRound {
            bound: 1,
            variables: 1
        })
    ));
    for degree in [9, usize::MAX - 1, usize::MAX] {
        assert!(matches!(
            round_polynomial(&[&[z; 2]], &[], degree, |_| z),
            Err(OracleError::DegreeOutOfRange { .. })
        ));
    }
}
