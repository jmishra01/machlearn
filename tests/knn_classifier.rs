//! Integration tests for the k-nearest-neighbors classifier.

use machlearn::{
    Dataset, KNeighborsClassifier, MlError, StratifiedKFold, Weighting, accuracy_score,
    cross_validate,
};
use ndarray::array;

#[test]
fn predicts_the_label_of_the_nearest_training_points() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )
    .unwrap();
    let model = KNeighborsClassifier::new(1).unwrap().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-2.5], [2.5]].view()).unwrap();

    assert_eq!(predictions, array!["no", "yes"]);
}

#[test]
fn matches_a_reference_solution() {
    // Reference predictions confirmed against
    // `sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)` fitted on the
    // same data, including its distance-tie resolution at the query `-0.5`
    // (equidistant between training rows at `-2.0` and `1.0`).
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )
    .unwrap();
    let model = KNeighborsClassifier::new(3).unwrap().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-0.5], [0.5], [2.5]].view()).unwrap();

    assert_eq!(predictions, array!["no", "yes", "yes"]);
}

#[test]
fn uniform_voting_breaks_ties_in_favor_of_the_smallest_sorted_label() {
    let dataset = Dataset::new(array![[-1.0], [3.0]], array!["b", "a"]).unwrap();
    let model = KNeighborsClassifier::new(2).unwrap().fit(&dataset).unwrap();

    let predictions = model.predict(array![[0.0]].view()).unwrap();

    assert_eq!(predictions, array!["a"]);
}

#[test]
fn distance_weighting_favors_the_closer_neighbor_over_a_uniform_tie() {
    let dataset = Dataset::new(array![[-1.0], [3.0]], array!["b", "a"]).unwrap();
    let query = array![[0.0]];

    let uniform = KNeighborsClassifier::new(2).unwrap().fit(&dataset).unwrap();
    let distance = KNeighborsClassifier::new(2)
        .unwrap()
        .with_weighting(Weighting::Distance)
        .fit(&dataset)
        .unwrap();

    assert_eq!(uniform.predict(query.view()).unwrap(), array!["a"]);
    assert_eq!(distance.predict(query.view()).unwrap(), array!["b"]);
}

#[test]
fn exposes_configuration_and_validates_neighbor_count() {
    let default = KNeighborsClassifier::default();
    assert_eq!(default.n_neighbors(), 5);
    assert_eq!(default.weighting(), Weighting::Uniform);

    let estimator = KNeighborsClassifier::new(3)
        .unwrap()
        .with_weighting(Weighting::Distance);
    assert_eq!(estimator.n_neighbors(), 3);
    assert_eq!(estimator.weighting(), Weighting::Distance);

    assert_eq!(
        KNeighborsClassifier::new(0).unwrap_err(),
        MlError::InvalidNeighborCount(0)
    );

    let dataset = Dataset::new(array![[-1.0], [1.0]], array!["no", "yes"]).unwrap();
    assert_eq!(
        KNeighborsClassifier::new(3)
            .unwrap()
            .fit(&dataset)
            .unwrap_err(),
        MlError::InsufficientSamples {
            required: 3,
            actual: 2,
        }
    );
}

#[test]
fn validates_prediction_features() {
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![0_u8, 1]).unwrap();
    let model = KNeighborsClassifier::new(1).unwrap().fit(&dataset).unwrap();

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
fn composes_with_stratified_cross_validation() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let labels = dataset.targets().to_vec();
    let folds = StratifiedKFold::new(3).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &KNeighborsClassifier::new(1).unwrap(),
        &dataset,
        &folds,
        accuracy_score,
    )
    .unwrap();

    assert_eq!(scores.scores(), &[1.0, 1.0, 1.0]);
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = KNeighborsClassifier::new(2)
        .unwrap()
        .with_weighting(Weighting::Distance);
    let dataset = Dataset::new(array![[-1.0], [1.0]], array!["no", "yes"]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: KNeighborsClassifier = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedKNeighborsClassifier<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.n_neighbors(), restored_model.n_neighbors());
    assert_eq!(model.weighting(), restored_model.weighting());
}
