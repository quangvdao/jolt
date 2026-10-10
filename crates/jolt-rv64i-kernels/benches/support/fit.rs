//! Least-squares calibration from fused-chain medians and a fixed-overhead control.

/// Affine fit of fused-chain medians, with the zero-length control kept separate.
pub struct ChainFit {
    pub slope: f64,
    pub intercept: f64,
    pub baseline: f64,
    pub reduction: f64,
    pub residual: f64,
}

impl ChainFit {
    pub fn new(points: &[(usize, f64)]) -> Option<Self> {
        if points.len() != 6
            || [0, 1, 2, 4, 8, 20]
                .iter()
                .any(|&n| points.iter().filter(|p| p.0 == n).count() != 1)
        {
            return None;
        }
        let baseline = points.iter().find(|p| p.0 == 0)?.1;
        let fitted: Vec<_> = points.iter().filter(|p| p.0 != 0).collect();
        if fitted.len() != 5 {
            return None;
        }
        let mean_x = fitted.iter().map(|p| p.0 as f64).sum::<f64>() / 5.0;
        let mean_y = fitted.iter().map(|p| p.1).sum::<f64>() / 5.0;
        let variance = fitted
            .iter()
            .map(|p| (p.0 as f64 - mean_x).powi(2))
            .sum::<f64>();
        let slope = fitted
            .iter()
            .map(|p| (p.0 as f64 - mean_x) * (p.1 - mean_y))
            .sum::<f64>()
            / variance;
        let intercept = mean_y - slope * mean_x;
        let residual = fitted
            .iter()
            .map(|p| (p.1 - (intercept + slope * p.0 as f64)).abs())
            .fold(0.0, f64::max);
        Some(Self {
            slope,
            intercept,
            baseline,
            reduction: intercept - baseline,
            residual,
        })
    }
}
