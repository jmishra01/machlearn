//! Integration tests for the bagged random-forest regressor.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, KFold, MaxFeatures, MlError, RandomForestRegressor, cross_validate, mean_squared_error,
};
use ndarray::{Array1, Array2, array};

/// A large, well-separated dataset (15 samples per cluster). With this many
/// samples per cluster, a bootstrap resample collapsing onto a single
/// cluster has negligible probability (`0.5^30`), so per-tree bagging noise
/// does not perturb predictions for interior query points.
fn separated_dataset() -> Dataset<f64> {
    let low = -10..5;
    let high = 5..20;
    let records: Vec<f64> = low.clone().chain(high.clone()).map(f64::from).collect();
    let targets: Vec<f64> = low.map(|_| 1.0).chain(high.map(|_| 10.0)).collect();
    let n_samples = records.len();
    Dataset::new(
        Array2::from_shape_vec((n_samples, 1), records).unwrap(),
        Array1::from_vec(targets),
    )
    .unwrap()
}

#[test]
fn predicts_close_to_the_cluster_means() {
    let model = RandomForestRegressor::new()
        .fit(&separated_dataset())
        .unwrap();

    let predictions = model.predict(array![[-8.0], [18.0]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 1.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(predictions[1], 10.0, epsilon = 1.0e-9);
}

#[test]
fn feature_importances_concentrate_on_the_informative_feature() {
    // `feature_1` is constant, so no split can ever use it: every tree's
    // importance for it is exactly zero, regardless of feature subsampling.
    let low = -10..5;
    let high = 5..20;
    let mut flat = Vec::new();
    for value in low.clone().chain(high.clone()) {
        flat.push(f64::from(value));
        flat.push(0.0);
    }
    let targets: Vec<f64> = low.map(|_| 1.0).chain(high.map(|_| 10.0)).collect();
    let n_samples = targets.len();
    let dataset = Dataset::new(
        Array2::from_shape_vec((n_samples, 2), flat).unwrap(),
        Array1::from_vec(targets),
    )
    .unwrap();

    let model = RandomForestRegressor::new()
        .with_max_features(MaxFeatures::All)
        .unwrap()
        .fit(&dataset)
        .unwrap();

    let importances = model.feature_importances();

    assert_abs_diff_eq!(importances[1], 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1.0e-9);
}

#[test]
fn is_deterministic_for_a_fixed_seed() {
    let estimator = RandomForestRegressor::new()
        .with_n_estimators(20)
        .unwrap()
        .with_seed(7);
    let first = estimator.fit(&separated_dataset()).unwrap();
    let second = estimator.fit(&separated_dataset()).unwrap();

    assert_eq!(first, second);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = RandomForestRegressor::default();
    assert_eq!(default.n_estimators(), 100);
    assert_eq!(default.max_depth(), None);
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);
    assert_eq!(default.max_features(), MaxFeatures::All);
    assert_eq!(default.seed(), 42);

    let estimator = RandomForestRegressor::new()
        .with_n_estimators(5)
        .unwrap()
        .with_max_depth(Some(3))
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap()
        .with_max_features(MaxFeatures::Log2)
        .unwrap()
        .with_seed(99);
    assert_eq!(estimator.n_estimators(), 5);
    assert_eq!(estimator.max_depth(), Some(3));
    assert_eq!(estimator.min_samples_split(), 4);
    assert_eq!(estimator.min_samples_leaf(), 2);
    assert_eq!(estimator.max_features(), MaxFeatures::Log2);
    assert_eq!(estimator.seed(), 99);

    assert_eq!(
        RandomForestRegressor::new()
            .with_n_estimators(0)
            .unwrap_err(),
        MlError::InvalidEstimatorCount(0)
    );
    assert_eq!(
        RandomForestRegressor::new()
            .with_max_features(MaxFeatures::Fixed(0))
            .unwrap_err(),
        MlError::InvalidMaxFeatures(0)
    );

    let non_finite_targets = Dataset::new(array![[0.0], [1.0]], array![0.0, f64::NAN]).unwrap();
    assert_eq!(
        RandomForestRegressor::new()
            .fit(&non_finite_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn validates_prediction_features() {
    let model = RandomForestRegressor::new()
        .fit(&separated_dataset())
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
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
    )
    .unwrap();
    let folds = KFold::new(3).unwrap().split(dataset.n_samples()).unwrap();

    let scores = cross_validate(
        &RandomForestRegressor::new().with_n_estimators(10).unwrap(),
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
    let estimator = RandomForestRegressor::new()
        .with_n_estimators(5)
        .unwrap()
        .with_seed(3);
    let dataset = separated_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: RandomForestRegressor = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedRandomForestRegressor =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    // Compared via predictions rather than `assert_eq!` on the whole struct:
    // `serde_json`'s text round trip for `f64` is not guaranteed bit-exact in
    // every case.
    let records = array![[-8.0], [18.0]];
    let original = model.predict(records.view()).unwrap();
    let restored = restored_model.predict(records.view()).unwrap();
    for (left, right) in original.iter().zip(restored.iter()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
