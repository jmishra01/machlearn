//! Integration tests for the k-nearest-neighbors regressor.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, KFold, KNeighborsRegressor, MlError, Weighting, cross_validate, mean_squared_error,
};
use ndarray::array;

#[test]
fn predicts_the_uniform_average_of_the_nearest_targets() {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0]],
        array![0.0, 1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let model = KNeighborsRegressor::new(3).unwrap().fit(&dataset).unwrap();

    let predictions = model.predict(array![[2.0]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 2.0, epsilon = 1.0e-12);
}

#[test]
fn matches_a_reference_solution() {
    // Reference predictions confirmed against
    // `sklearn.neighbors.KNeighborsRegressor(n_neighbors=2)` (both `uniform`
    // and `distance` weights) fitted on the same data.
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0]],
        array![0.0, 1.0, 4.0, 9.0, 16.0],
    )
    .unwrap();
    let query = array![[0.5], [2.5], [3.5]];

    let uniform = KNeighborsRegressor::new(2).unwrap().fit(&dataset).unwrap();
    let predictions = uniform.predict(query.view()).unwrap();
    assert_abs_diff_eq!(predictions[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(predictions[1], 6.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(predictions[2], 12.5, epsilon = 1.0e-12);

    let distance = KNeighborsRegressor::new(2)
        .unwrap()
        .with_weighting(Weighting::Distance)
        .fit(&dataset)
        .unwrap();
    let distance_predictions = distance.predict(query.view()).unwrap();
    assert_abs_diff_eq!(distance_predictions[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(distance_predictions[1], 6.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(distance_predictions[2], 12.5, epsilon = 1.0e-12);
}

#[test]
fn distance_weighting_favors_the_closer_neighbor_over_a_uniform_average() {
    let dataset = Dataset::new(array![[-1.0], [3.0]], array![10.0, 30.0]).unwrap();
    let query = array![[0.0]];

    let uniform = KNeighborsRegressor::new(2).unwrap().fit(&dataset).unwrap();
    let distance = KNeighborsRegressor::new(2)
        .unwrap()
        .with_weighting(Weighting::Distance)
        .fit(&dataset)
        .unwrap();

    let uniform_prediction = uniform.predict(query.view()).unwrap()[0];
    let distance_prediction = distance.predict(query.view()).unwrap()[0];

    assert_abs_diff_eq!(uniform_prediction, 20.0, epsilon = 1.0e-12);
    // Distance weighting pulls the prediction toward the closer point (10.0
    // at distance 1) rather than the midpoint the uniform average produces.
    assert!(distance_prediction < uniform_prediction);
}

#[test]
fn exact_match_receives_the_entire_distance_weight() {
    let dataset = Dataset::new(array![[0.0], [10.0]], array![5.0, 50.0]).unwrap();
    let model = KNeighborsRegressor::new(2)
        .unwrap()
        .with_weighting(Weighting::Distance)
        .fit(&dataset)
        .unwrap();

    let prediction = model.predict(array![[0.0]].view()).unwrap()[0];

    assert_abs_diff_eq!(prediction, 5.0, epsilon = 1.0e-12);
}

#[test]
fn exposes_configuration_and_validates_neighbor_count() {
    let default = KNeighborsRegressor::default();
    assert_eq!(default.n_neighbors(), 5);
    assert_eq!(default.weighting(), Weighting::Uniform);

    let estimator = KNeighborsRegressor::new(3)
        .unwrap()
        .with_weighting(Weighting::Distance);
    assert_eq!(estimator.n_neighbors(), 3);
    assert_eq!(estimator.weighting(), Weighting::Distance);

    assert_eq!(
        KNeighborsRegressor::new(0).unwrap_err(),
        MlError::InvalidNeighborCount(0)
    );

    let dataset = Dataset::new(array![[0.0], [1.0]], array![0.0, 1.0]).unwrap();
    assert_eq!(
        KNeighborsRegressor::new(3)
            .unwrap()
            .fit(&dataset)
            .unwrap_err(),
        MlError::InsufficientSamples {
            required: 3,
            actual: 2,
        }
    );

    let non_finite_targets = Dataset::new(array![[0.0], [1.0]], array![0.0, f64::NAN]).unwrap();
    assert_eq!(
        KNeighborsRegressor::new(1)
            .unwrap()
            .fit(&non_finite_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn validates_prediction_features() {
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![-1.0, 1.0]).unwrap();
    let model = KNeighborsRegressor::new(1).unwrap().fit(&dataset).unwrap();

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
        &KNeighborsRegressor::new(1).unwrap(),
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
    let estimator = KNeighborsRegressor::new(2)
        .unwrap()
        .with_weighting(Weighting::Distance);
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![-1.0, 1.0]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: KNeighborsRegressor = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedKNeighborsRegressor =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    // Compared via predictions rather than `assert_eq!` on the whole struct:
    // `serde_json`'s text round trip for `f64` is not guaranteed bit-exact in
    // every case.
    let records = array![[-1.0], [1.0]];
    let original = model.predict(records.view()).unwrap();
    let restored = restored_model.predict(records.view()).unwrap();
    for (left, right) in original.iter().zip(restored.iter()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
