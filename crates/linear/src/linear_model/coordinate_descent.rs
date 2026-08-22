use ndarray::{Array1, Axis};

use crate::linear_model::{convergence::ConvergenceReport, ridge_regression::center};
use machlearn_core::core::{Dataset, MlError, Result, validate_features};

const MINIMUM_FEATURE_NORM: f64 = 1.0e-12;

/// Fits an L1/L2-penalized linear model by coordinate descent.
///
/// Minimizes the sum of the mean-squared error, an L1 penalty scaled by
/// `alpha * l1_ratio`, and an L2 penalty scaled by `0.5 * alpha * (1 -
/// l1_ratio)`, cycling through each coefficient in turn and applying soft
/// thresholding. `l1_ratio = 1.0` gives Lasso; `l1_ratio = 0.0` gives
/// coordinate-descent Ridge (for which
/// [`crate::linear_model::RidgeRegression`]'s closed-form solver is exact
/// and preferred).
///
/// Unlike the QR-based solvers in this module, this accepts more features
/// than samples: coordinate descent does not require full column rank.
///
/// Convergence is measured by the largest coefficient change in a full pass
/// over every feature; iteration stops once that change falls to or below
/// `tolerance`.
///
/// # Errors
///
/// Returns an error when features are empty or non-finite, when a target is
/// non-finite, or when the solver fails to converge within
/// `max_iterations`.
pub(super) fn fit_coordinate_descent(
    dataset: &Dataset<f64>,
    fit_intercept: bool,
    alpha: f64,
    l1_ratio: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Array1<f64>, f64, ConvergenceReport)> {
    validate_features(dataset.records())?;

    let (centered_records, centered_targets, feature_means, target_mean) = if fit_intercept {
        center(dataset)?
    } else {
        for (index, &target) in dataset.targets().iter().enumerate() {
            if !target.is_finite() {
                return Err(MlError::NonFiniteActualTarget { index });
            }
        }
        (
            dataset.records().to_owned(),
            dataset.targets().to_owned(),
            Array1::zeros(dataset.n_features()),
            0.0,
        )
    };

    #[allow(clippy::cast_precision_loss)]
    let n_samples = dataset.n_samples() as f64;
    let l1_penalty = alpha * l1_ratio;
    let l2_penalty = alpha * (1.0 - l1_ratio);

    let feature_norms: Vec<f64> = centered_records
        .axis_iter(Axis(1))
        .map(|column| column.iter().map(|value| value * value).sum::<f64>() / n_samples)
        .collect();

    let mut coefficients = Array1::<f64>::zeros(dataset.n_features());
    let mut residual = centered_targets;

    let mut iterations = 0;
    let mut max_change = 0.0_f64;
    let mut converged = false;
    for _iteration in 0..max_iterations {
        iterations += 1;
        max_change = 0.0;
        for (feature_index, &feature_norm) in feature_norms.iter().enumerate() {
            if feature_norm <= MINIMUM_FEATURE_NORM {
                continue;
            }
            let column = centered_records.column(feature_index);
            let previous_coefficient = coefficients[feature_index];

            residual.scaled_add(previous_coefficient, &column);
            let rho = column.dot(&residual) / n_samples;
            let new_coefficient = soft_threshold(rho, l1_penalty) / (feature_norm + l2_penalty);
            residual.scaled_add(-new_coefficient, &column);

            max_change = max_change.max((new_coefficient - previous_coefficient).abs());
            coefficients[feature_index] = new_coefficient;
        }
        if max_change <= tolerance {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(MlError::OptimizationDidNotConverge {
            iterations: max_iterations,
        });
    }

    let intercept = target_mean - feature_means.dot(&coefficients);
    if !intercept.is_finite() {
        return Err(MlError::NonFiniteSolverOutput {
            index: coefficients.len(),
        });
    }
    if let Some((index, _coefficient)) = coefficients
        .iter()
        .enumerate()
        .find(|(_index, coefficient)| !coefficient.is_finite())
    {
        return Err(MlError::NonFiniteSolverOutput { index });
    }

    Ok((
        coefficients,
        intercept,
        ConvergenceReport {
            iterations,
            max_parameter_change: max_change,
            tolerance,
        },
    ))
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

pub(super) fn validate_l1_ratio(l1_ratio: f64) -> Result<()> {
    if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) {
        return Err(MlError::InvalidL1Ratio(l1_ratio));
    }
    Ok(())
}
