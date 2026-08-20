//! Integration tests for Lasso (L1-regularized) linear regression.

use approx::assert_abs_diff_eq;
use machlearn::{Dataset, KFold, LassoRegression, MlError, cross_validate, mean_squared_error};
use ndarray::array;

fn correlated_dataset() -> Dataset<f64> {
    Dataset::new(
        array![
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 2.0],
            [5.0, 0.0, 0.0],
            [6.0, 1.0, 4.0],
        ],
        array![3.1, 4.8, 7.05, 9.0, 10.9, 13.15],
    )
    .unwrap()
}

#[test]
fn matches_a_reference_solution() {
    // Reference values confirmed against `sklearn.linear_model.Lasso(alpha=0.5,
    // tol=1e-10, max_iter=100000)` fitted on the same data. Two of the three
    // features are shrunk to exactly zero, matching Lasso's sparsity.
    let model = LassoRegression::new(0.5)
        .unwrap()
        .with_tolerance(1.0e-10)
        .unwrap()
        .with_max_iterations(100_000)
        .unwrap()
        .fit(&correlated_dataset())
        .unwrap();

    assert_abs_diff_eq!(
        model.coefficients()[0],
        1.842_857_142_857_143,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(model.coefficients()[1], 0.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.coefficients()[2], 0.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.intercept(), 1.55, epsilon = 1.0e-8);
    assert_eq!(model.n_nonzero_coefficients(), 1);

    let prediction = model.predict(array![[2.5, 0.5, 1.0]].view()).unwrap();
    assert_abs_diff_eq!(prediction[0], 6.157_142_857_142_857, epsilon = 1.0e-8);
}

#[test]
fn fits_when_there_are_more_features_than_samples() {
    // Coordinate descent, unlike the QR-based solvers, does not require
    // `n_samples >= n_features`.
    let dataset = Dataset::new(
        array![[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0]],
        array![5.0, 3.0],
    )
    .unwrap();

    let model = LassoRegression::new(0.1).unwrap().fit(&dataset).unwrap();

    assert_eq!(model.n_features(), 4);
    let predictions = model.predict(dataset.records()).unwrap();
    assert!(predictions.iter().all(|value| value.is_finite()));
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let estimator = LassoRegression::new(0.5)
        .unwrap()
        .with_intercept(false)
        .with_max_iterations(500)
        .unwrap()
        .with_tolerance(1.0e-6)
        .unwrap();
    assert_abs_diff_eq!(estimator.alpha(), 0.5);
    assert!(!estimator.fit_intercept());
    assert_eq!(estimator.max_iterations(), 500);
    assert_abs_diff_eq!(estimator.tolerance(), 1.0e-6);

    let default = LassoRegression::new(1.0).unwrap();
    assert_eq!(default.max_iterations(), 1000);
    assert_abs_diff_eq!(default.tolerance(), 1.0e-4);
    assert!(default.fit_intercept());

    assert_eq!(
        LassoRegression::new(-1.0).unwrap_err(),
        MlError::InvalidRegularization(-1.0)
    );
    assert_eq!(
        LassoRegression::new(1.0)
            .unwrap()
            .with_max_iterations(0)
            .unwrap_err(),
        MlError::InvalidMaxIterations(0)
    );
    assert_eq!(
        LassoRegression::new(1.0)
            .unwrap()
            .with_tolerance(0.0)
            .unwrap_err(),
        MlError::InvalidTolerance(0.0)
    );
}

#[test]
fn reports_non_convergence_within_a_tiny_iteration_budget() {
    let estimator = LassoRegression::new(0.5)
        .unwrap()
        .with_max_iterations(1)
        .unwrap()
        .with_tolerance(1.0e-12)
        .unwrap();

    assert_eq!(
        estimator.fit(&correlated_dataset()).unwrap_err(),
        MlError::OptimizationDidNotConverge { iterations: 1 }
    );
}

#[test]
fn validates_prediction_features_and_targets() {
    let model = LassoRegression::new(1.0)
        .unwrap()
        .fit(&correlated_dataset())
        .unwrap();

    assert_eq!(
        model.predict(array![[1.0, 2.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 3,
            actual: 2,
        }
    );
    assert_eq!(
        model
            .predict(array![[f64::NAN, 0.0, 0.0]].view())
            .unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );

    let non_finite_targets = Dataset::new(array![[0.0], [1.0]], array![0.0, f64::NAN]).unwrap();
    assert_eq!(
        LassoRegression::new(1.0)
            .unwrap()
            .fit(&non_finite_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn composes_with_cross_validation() {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
    )
    .unwrap();
    let folds = KFold::new(3).unwrap().split(dataset.n_samples()).unwrap();

    let scores = cross_validate(
        &LassoRegression::new(0.01).unwrap(),
        &dataset,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert_eq!(scores.scores().len(), 3);
    assert!(scores.scores().iter().all(|&score| score >= 0.0));
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = LassoRegression::new(0.5).unwrap().with_intercept(false);
    let dataset = correlated_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: LassoRegression = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedLassoRegression =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    let predictions = model.predict(dataset.records()).unwrap();
    let restored_predictions = restored_model.predict(dataset.records()).unwrap();
    for (left, right) in predictions.iter().zip(restored_predictions.iter()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
