//! Integration tests for the gradient-boosted decision-tree regressor.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, GradientBoostingRegressor, KFold, MlError, cross_validate, mean_squared_error,
};
use ndarray::array;

fn dataset() -> Dataset<f64> {
    Dataset::new(
        array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ],
        array![1.0, 1.2, 1.5, 4.8, 5.1, 5.3, 9.9, 10.1, 10.4, 10.0],
    )
    .unwrap()
}

#[test]
fn matches_sklearn_gradient_boosting_regressor() {
    // Reference values confirmed against
    // `sklearn.ensemble.GradientBoostingRegressor(n_estimators=20,
    // learning_rate=0.1, max_depth=3, random_state=0)` fitted on the same
    // data. `random_state` only affects sklearn's internal split-tie
    // sampling; the deterministic tree growth here matches exactly.
    let model = GradientBoostingRegressor::new()
        .with_n_estimators(20)
        .unwrap()
        .fit(&dataset())
        .unwrap();

    let predictions = model.predict(array![[0.5], [3.5], [8.5]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 1.623_468_178_119_929_4, epsilon = 1.0e-9);
    assert_abs_diff_eq!(predictions[1], 4.954_748_573_166_578, epsilon = 1.0e-9);
    assert_abs_diff_eq!(predictions[2], 9.852_274_656_874_192, epsilon = 1.0e-9);
}

#[test]
fn matches_sklearn_with_custom_tree_shape_parameters() {
    // Reference values confirmed against
    // `sklearn.ensemble.GradientBoostingRegressor(n_estimators=10,
    // learning_rate=0.2, max_depth=2, min_samples_split=4,
    // min_samples_leaf=2, random_state=0)` fitted on the same data.
    let model = GradientBoostingRegressor::new()
        .with_n_estimators(10)
        .unwrap()
        .with_learning_rate(0.2)
        .unwrap()
        .with_max_depth(Some(2))
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap()
        .fit(&dataset())
        .unwrap();

    let predictions = model.predict(array![[0.5], [3.5], [8.5]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 1.723_659_83, epsilon = 1.0e-6);
    assert_abs_diff_eq!(predictions[1], 5.028_505_65, epsilon = 1.0e-6);
    assert_abs_diff_eq!(predictions[2], 9.745_056, epsilon = 1.0e-6);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = GradientBoostingRegressor::default();
    assert_eq!(default.n_estimators(), 100);
    assert_abs_diff_eq!(default.learning_rate(), 0.1);
    assert_eq!(default.max_depth(), Some(3));
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);

    let estimator = GradientBoostingRegressor::new()
        .with_n_estimators(5)
        .unwrap()
        .with_learning_rate(0.5)
        .unwrap()
        .with_max_depth(None)
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap();
    assert_eq!(estimator.n_estimators(), 5);
    assert_abs_diff_eq!(estimator.learning_rate(), 0.5);
    assert_eq!(estimator.max_depth(), None);
    assert_eq!(estimator.min_samples_split(), 4);
    assert_eq!(estimator.min_samples_leaf(), 2);

    assert_eq!(
        GradientBoostingRegressor::new()
            .with_n_estimators(0)
            .unwrap_err(),
        MlError::InvalidEstimatorCount(0)
    );
    assert_eq!(
        GradientBoostingRegressor::new()
            .with_learning_rate(0.0)
            .unwrap_err(),
        MlError::InvalidLearningRate(0.0)
    );
    assert!(matches!(
        GradientBoostingRegressor::new()
            .with_learning_rate(f64::NAN)
            .unwrap_err(),
        MlError::InvalidLearningRate(value) if value.is_nan()
    ));

    let non_finite_targets = Dataset::new(array![[0.0], [1.0]], array![0.0, f64::NAN]).unwrap();
    assert_eq!(
        GradientBoostingRegressor::new()
            .fit(&non_finite_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn validates_prediction_features() {
    let model = GradientBoostingRegressor::new()
        .with_n_estimators(5)
        .unwrap()
        .fit(&dataset())
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
}

#[test]
fn composes_with_cross_validation() {
    let folds = KFold::new(5).unwrap().split(dataset().n_samples()).unwrap();

    let scores = cross_validate(
        &GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .unwrap(),
        &dataset(),
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert_eq!(scores.scores().len(), 5);
    assert!(scores.scores().iter().all(|&score| score >= 0.0));
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = GradientBoostingRegressor::new()
        .with_n_estimators(5)
        .unwrap();
    let model = estimator.fit(&dataset()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: GradientBoostingRegressor =
        serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedGradientBoostingRegressor =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    // Compared via predictions rather than `assert_eq!` on the whole struct:
    // `serde_json`'s text round trip for `f64` is not guaranteed bit-exact in
    // every case.
    let records = array![[0.5], [8.5]];
    let original = model.predict(records.view()).unwrap();
    let restored = restored_model.predict(records.view()).unwrap();
    for (left, right) in original.iter().zip(restored.iter()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
