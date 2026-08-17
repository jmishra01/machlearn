//! Integration tests for regression metrics.

use approx::assert_abs_diff_eq;
use machlearn::{
    MlError, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
};
use ndarray::{Array1, array};

#[test]
fn matches_reference_regression_values() {
    let actual = array![3.0, -0.5, 2.0, 7.0];
    let predicted = array![2.5, 0.0, 2.0, 8.0];

    assert_abs_diff_eq!(
        mean_squared_error(actual.view(), predicted.view()).unwrap(),
        0.375,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        root_mean_squared_error(actual.view(), predicted.view()).unwrap(),
        0.375_f64.sqrt(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        mean_absolute_error(actual.view(), predicted.view()).unwrap(),
        0.5,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        r2_score(actual.view(), predicted.view()).unwrap(),
        0.948_608_137_044_967_9,
        epsilon = 1.0e-12
    );
}

#[test]
fn perfect_predictions_have_zero_error_and_unit_r2() {
    let values = array![-2.0, 0.0, 4.0];

    assert_abs_diff_eq!(
        mean_squared_error(values.view(), values.view()).unwrap(),
        0.0
    );
    assert_abs_diff_eq!(
        root_mean_squared_error(values.view(), values.view()).unwrap(),
        0.0
    );
    assert_abs_diff_eq!(
        mean_absolute_error(values.view(), values.view()).unwrap(),
        0.0
    );
    assert_abs_diff_eq!(r2_score(values.view(), values.view()).unwrap(), 1.0);
}

#[test]
fn rejects_empty_inputs() {
    let empty = Array1::<f64>::zeros(0);
    assert_eq!(
        mean_squared_error(empty.view(), empty.view()).unwrap_err(),
        MlError::EmptyMetricInput
    );
}

#[test]
fn rejects_different_input_lengths() {
    assert_eq!(
        mean_absolute_error(array![1.0, 2.0].view(), array![1.0].view()).unwrap_err(),
        MlError::MismatchedMetricInput {
            actual: 2,
            predicted: 1,
        }
    );
}

#[test]
fn identifies_non_finite_input_positions() {
    assert_eq!(
        mean_squared_error(array![1.0, f64::NAN].view(), array![1.0, 2.0].view()).unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
    assert_eq!(
        mean_squared_error(array![1.0, 2.0].view(), array![1.0, f64::INFINITY].view()).unwrap_err(),
        MlError::NonFinitePrediction { index: 1 }
    );
}

#[test]
fn r2_rejects_constant_targets() {
    assert_eq!(
        r2_score(array![2.0, 2.0].view(), array![2.0, 2.0].view()).unwrap_err(),
        MlError::ConstantTargets
    );
}

#[test]
fn reports_non_finite_results_from_extreme_finite_inputs() {
    assert_eq!(
        mean_squared_error(array![f64::MAX].view(), array![-f64::MAX].view()).unwrap_err(),
        MlError::NonFiniteMetricResult {
            metric: "mean_squared_error"
        }
    );
}
