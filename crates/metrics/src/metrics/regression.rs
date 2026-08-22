// `ArrayView1` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::ArrayView1;

use machlearn_core::core::{MlError, Result};

/// Returns the mean squared difference between actual and predicted values.
///
/// # Errors
///
/// Returns an error for empty, different-length, or non-finite inputs, or when
/// the computation produces a non-finite result.
pub fn mean_squared_error(
    actual: ArrayView1<'_, f64>,
    predicted: ArrayView1<'_, f64>,
) -> Result<f64> {
    validate_inputs(actual, predicted)?;
    #[allow(clippy::cast_precision_loss)]
    let count = actual.len() as f64;
    let result = actual
        .iter()
        .zip(predicted)
        .map(|(&actual, &predicted)| {
            let difference = actual - predicted;
            (difference * difference) / count
        })
        .sum();
    finite_result("mean_squared_error", result)
}

/// Returns the square root of the mean squared error.
///
/// # Errors
///
/// Returns an error for empty, different-length, or non-finite inputs, or when
/// the computation produces a non-finite result.
pub fn root_mean_squared_error(
    actual: ArrayView1<'_, f64>,
    predicted: ArrayView1<'_, f64>,
) -> Result<f64> {
    let result = mean_squared_error(actual, predicted)?.sqrt();
    finite_result("root_mean_squared_error", result)
}

/// Returns the mean absolute difference between actual and predicted values.
///
/// # Errors
///
/// Returns an error for empty, different-length, or non-finite inputs, or when
/// the computation produces a non-finite result.
pub fn mean_absolute_error(
    actual: ArrayView1<'_, f64>,
    predicted: ArrayView1<'_, f64>,
) -> Result<f64> {
    validate_inputs(actual, predicted)?;
    #[allow(clippy::cast_precision_loss)]
    let count = actual.len() as f64;
    let result = actual
        .iter()
        .zip(predicted)
        .map(|(&actual, &predicted)| (actual - predicted).abs() / count)
        .sum();
    finite_result("mean_absolute_error", result)
}

/// Returns the coefficient of determination, commonly called R-squared.
///
/// A value of one represents perfect predictions. Values may be negative when
/// predictions perform worse than always predicting the actual-target mean.
///
/// # Errors
///
/// Returns an error for empty, different-length, or non-finite inputs. It also
/// returns [`MlError::ConstantTargets`] when R-squared is undefined because all
/// actual targets are identical, and an error for non-finite intermediate or
/// final results.
pub fn r2_score(actual: ArrayView1<'_, f64>, predicted: ArrayView1<'_, f64>) -> Result<f64> {
    validate_inputs(actual, predicted)?;
    #[allow(clippy::cast_precision_loss)]
    let count = actual.len() as f64;
    let actual_mean = finite_result("r2_score", actual.iter().map(|value| value / count).sum())?;

    let residual_sum: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(&actual, &predicted)| {
            let difference = actual - predicted;
            difference * difference
        })
        .sum();
    let total_sum: f64 = actual
        .iter()
        .map(|&actual| {
            let difference = actual - actual_mean;
            difference * difference
        })
        .sum();
    finite_result("r2_score", residual_sum)?;
    finite_result("r2_score", total_sum)?;

    if total_sum == 0.0 {
        return Err(MlError::ConstantTargets);
    }
    finite_result("r2_score", 1.0 - residual_sum / total_sum)
}

fn validate_inputs(actual: ArrayView1<'_, f64>, predicted: ArrayView1<'_, f64>) -> Result<()> {
    if actual.is_empty() {
        return Err(MlError::EmptyMetricInput);
    }
    if actual.len() != predicted.len() {
        return Err(MlError::MismatchedMetricInput {
            actual: actual.len(),
            predicted: predicted.len(),
        });
    }
    if let Some(index) = actual.iter().position(|value| !value.is_finite()) {
        return Err(MlError::NonFiniteActualTarget { index });
    }
    if let Some(index) = predicted.iter().position(|value| !value.is_finite()) {
        return Err(MlError::NonFinitePrediction { index });
    }
    Ok(())
}

fn finite_result(metric: &'static str, result: f64) -> Result<f64> {
    if !result.is_finite() {
        return Err(MlError::NonFiniteMetricResult { metric });
    }
    Ok(result)
}
