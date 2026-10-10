#![cfg(feature = "test-utils")]

#[path = "../benches/support/fit.rs"]
mod fit;

use fit::ChainFit;

#[expect(
    clippy::unwrap_used,
    reason = "a valid frozen affine calibration must fit"
)]
#[test]
fn affine_calibration_recovers_slope_intercept_and_reduction() {
    let fit =
        ChainFit::new(&[(0, 0.8), (1, 1.75), (2, 2.0), (4, 2.5), (8, 3.5), (20, 6.5)]).unwrap();
    for (actual, expected) in [
        (fit.slope, 0.25),
        (fit.intercept, 1.5),
        (fit.baseline, 0.8),
        (fit.reduction, 0.7),
        (fit.residual, 0.0),
    ] {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(
        ChainFit::new(&[(0, 0.8), (1, 1.75), (2, 2.0), (4, 2.5), (8, 3.5), (8, 3.5)]).is_none()
    );
}
