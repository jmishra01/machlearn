//! Integration tests for L2-regularized linear regression.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, KFold, LinearRegression, MlError, RidgeRegression, cross_validate, mean_squared_error,
};
use ndarray::array;

#[test]
fn matches_a_closed_form_solution_without_an_intercept() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![1.0, 2.0, 3.0]).unwrap();

    let model = RidgeRegression::new(1.0)
        .unwrap()
        .with_intercept(false)
        .fit(&dataset)
        .unwrap();

    assert_abs_diff_eq!(model.coefficients()[0], 14.0 / 15.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.intercept(), 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.alpha(), 1.0, epsilon = 1.0e-12);
    assert_eq!(model.n_features(), 1);
}

#[test]
fn fits_an_unpenalized_intercept() {
    let dataset = Dataset::new(array![[0.0], [1.0], [2.0]], array![1.0, 3.0, 5.0]).unwrap();

    let model = RidgeRegression::new(1.0).unwrap().fit(&dataset).unwrap();

    assert_abs_diff_eq!(model.coefficients()[0], 4.0 / 3.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.intercept(), 5.0 / 3.0, epsilon = 1.0e-12);
}

#[test]
fn zero_regularization_matches_ordinary_least_squares() {
    let dataset = Dataset::new(
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [3.0, -1.0]],
        array![3.0, 5.0, 2.0, 6.5, 9.5],
    )
    .unwrap();

    let ordinary = LinearRegression::new().fit(&dataset).unwrap();
    let ridge = RidgeRegression::new(0.0).unwrap().fit(&dataset).unwrap();

    assert_abs_diff_eq!(ridge.intercept(), ordinary.intercept(), epsilon = 1.0e-12);
    for (ridge, ordinary) in ridge.coefficients().iter().zip(ordinary.coefficients()) {
        assert_abs_diff_eq!(ridge, ordinary, epsilon = 1.0e-12);
    }
}

#[test]
fn regularization_supports_underdetermined_collinear_data() {
    let dataset = Dataset::new(array![[1.0, 2.0]], array![3.0]).unwrap();

    let model = RidgeRegression::new(1.0)
        .unwrap()
        .with_intercept(false)
        .fit(&dataset)
        .unwrap();

    assert_abs_diff_eq!(model.coefficients()[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.coefficients()[1], 1.0, epsilon = 1.0e-12);
}

#[test]
fn validates_regularization_and_exposes_configuration() {
    let estimator = RidgeRegression::new(2.5).unwrap().with_intercept(false);
    assert_abs_diff_eq!(estimator.alpha(), 2.5);
    assert!(!estimator.fit_intercept());
    assert_abs_diff_eq!(RidgeRegression::default().alpha(), 1.0);
    assert!(RidgeRegression::default().fit_intercept());

    assert_eq!(
        RidgeRegression::new(-1.0).unwrap_err(),
        MlError::InvalidRegularization(-1.0)
    );
    assert_eq!(
        RidgeRegression::new(f64::INFINITY).unwrap_err(),
        MlError::InvalidRegularization(f64::INFINITY)
    );
    assert!(matches!(
        RidgeRegression::new(f64::NAN),
        Err(MlError::InvalidRegularization(value)) if value.is_nan()
    ));
}

#[test]
fn validates_targets_features_and_prediction_outputs() {
    let invalid_targets = Dataset::new(array![[1.0], [2.0]], array![1.0, f64::NAN]).unwrap();
    assert_eq!(
        RidgeRegression::new(1.0)
            .unwrap()
            .fit(&invalid_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );

    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = RidgeRegression::new(1.0)
        .unwrap()
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
        &RidgeRegression::new(0.1).unwrap(),
        &dataset,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert_eq!(scores.scores().len(), 3);
    assert!(scores.scores().iter().all(|score| score.is_finite()));
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = RidgeRegression::new(0.5).unwrap().with_intercept(false);
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: RidgeRegression = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedRidgeRegression =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model, restored_model);
}
