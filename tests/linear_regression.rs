//! Integration tests for ordinary least-squares regression.

use approx::assert_abs_diff_eq;
use machlearn::{Dataset, KFold, LinearRegression, MlError, cross_validate, mean_squared_error};
use ndarray::array;

#[test]
fn fits_a_multivariate_reference_solution() {
    let dataset = Dataset::new(
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [3.0, -1.0]],
        array![3.0, 5.0, 2.0, 6.5, 9.5],
    )
    .unwrap();

    let model = LinearRegression::new().fit(&dataset).unwrap();

    assert_abs_diff_eq!(model.intercept(), 3.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.coefficients()[0], 2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.coefficients()[1], -0.5, epsilon = 1.0e-12);
    assert_eq!(model.n_features(), 2);

    let predictions = model
        .predict(array![[4.0, 2.0], [-1.0, 4.0]].view())
        .unwrap();
    assert_abs_diff_eq!(predictions[0], 10.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(predictions[1], -1.0, epsilon = 1.0e-12);
}

#[test]
fn supports_regression_without_an_intercept() {
    let dataset = Dataset::new(
        array![[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0]],
        array![2.0, -1.0, 3.0, -1.0],
    )
    .unwrap();
    let estimator = LinearRegression::new().with_intercept(false);

    let model = estimator.fit(&dataset).unwrap();

    assert!(!estimator.fit_intercept());
    assert_abs_diff_eq!(model.intercept(), 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.coefficients()[0], 2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.coefficients()[1], -1.0, epsilon = 1.0e-12);
}

#[test]
fn defaults_to_fitting_an_intercept() {
    assert!(LinearRegression::default().fit_intercept());
    assert!(LinearRegression::new().fit_intercept());
}

#[test]
fn rejects_insufficient_samples_and_rank_deficiency() {
    let insufficient = Dataset::new(array![[1.0, 2.0], [3.0, 4.0]], array![1.0, 2.0]).unwrap();
    assert_eq!(
        LinearRegression::new().fit(&insufficient).unwrap_err(),
        MlError::InsufficientSamples {
            required: 3,
            actual: 2,
        }
    );

    let collinear = Dataset::new(
        array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]],
        array![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    assert_eq!(
        LinearRegression::new().fit(&collinear).unwrap_err(),
        MlError::RankDeficientDesign {
            rank: 2,
            feature_count: 3,
        }
    );
}

#[test]
fn rejects_non_finite_targets() {
    let dataset = Dataset::new(array![[1.0], [2.0]], array![1.0, f64::NAN]).unwrap();

    assert_eq!(
        LinearRegression::new().fit(&dataset).unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn validates_prediction_features_and_outputs() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = LinearRegression::new()
        .with_intercept(false)
        .fit(&dataset)
        .unwrap();

    assert_eq!(
        model.predict(array![[1.0, 2.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 1,
            actual: 2,
        }
    );
    assert_eq!(
        model.predict(array![[f64::NAN]].view()).unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
    assert_eq!(
        model.predict(array![[f64::MAX]].view()).unwrap_err(),
        MlError::NonFinitePrediction { index: 0 }
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
        &LinearRegression::new(),
        &dataset,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert!(scores.scores().iter().all(|score| *score < 1.0e-24));
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = LinearRegression::new().with_intercept(false);
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: LinearRegression = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedLinearRegression =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model, restored_model);
}
