//! Integration tests for the `AdaBoost` (discrete SAMME) binary classifier.

use approx::assert_abs_diff_eq;
use machlearn::{
    AdaBoostClassifier, Dataset, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn separable_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "no", "yes", "yes", "yes", "yes"],
    )
    .unwrap()
}

fn noisy_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![
            [-3.0],
            [-2.0],
            [-1.0],
            [-0.5],
            [0.4],
            [0.6],
            [1.0],
            [1.5],
            [2.0],
            [3.0]
        ],
        array![
            "no", "no", "no", "yes", "no", "yes", "yes", "no", "yes", "yes"
        ],
    )
    .unwrap()
}

#[test]
fn a_perfect_first_stump_short_circuits_the_ensemble() {
    // Reference values confirmed against
    // `sklearn.ensemble.AdaBoostClassifier(n_estimators=10,
    // learning_rate=1.0, random_state=0)` fitted on the same data: a
    // trivially separable dataset lets the very first stump classify
    // perfectly, so sklearn (and this implementation) stop after one
    // round with a voting weight of exactly one, giving `sigmoid(±2)`
    // probabilities.
    let model = AdaBoostClassifier::new()
        .with_n_estimators(10)
        .unwrap()
        .fit(&separable_dataset())
        .unwrap();

    assert_eq!(model.n_estimators(), 1);

    let query = array![[-0.25], [0.25]];
    let predictions = model.predict(query.view()).unwrap();
    assert_eq!(predictions, array!["no", "yes"]);

    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.880_797_08, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.119_202_92, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.119_202_92, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.880_797_08, epsilon = 1.0e-6);
}

#[test]
fn matches_sklearn_over_multiple_boosting_rounds() {
    // Reference values confirmed against
    // `sklearn.ensemble.AdaBoostClassifier(n_estimators=6,
    // learning_rate=1.0, random_state=0)` fitted on the same data, which
    // requires the full six rounds (no early stop) since no single stump
    // classifies it perfectly.
    let model = AdaBoostClassifier::new()
        .with_n_estimators(6)
        .unwrap()
        .fit(&noisy_dataset())
        .unwrap();

    assert_eq!(model.n_estimators(), 6);

    let query = array![[-0.25], [0.25], [1.2]];
    let predictions = model.predict(query.view()).unwrap();
    assert_eq!(predictions, array!["yes", "no", "yes"]);

    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.611_415_89, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.445_753_64, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[2, 1]], 0.584_789_36, epsilon = 1.0e-6);
}

#[test]
fn runs_the_full_configured_round_count_when_no_round_stops_early() {
    // Reference confirmed against `sklearn.ensemble.AdaBoostClassifier
    // (n_estimators=100, learning_rate=1.0, random_state=0)`: on this
    // dataset no round ever reaches a perfect or worse-than-random weak
    // learner, so boosting runs every configured round.
    let model = AdaBoostClassifier::new()
        .with_n_estimators(100)
        .unwrap()
        .fit(&noisy_dataset())
        .unwrap();

    assert_eq!(model.n_estimators(), 100);
}

#[test]
fn rejects_a_first_round_no_better_than_random() {
    // A single-axis stump cannot beat 50% weighted error on this
    // checkerboard-style dataset; sklearn raises the analogous "ensemble
    // can not be fit" error for the same input.
    let dataset = Dataset::new(
        array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.9],
        ],
        array!["a", "b", "b", "a", "a", "b", "b", "a"],
    )
    .unwrap();

    assert_eq!(
        AdaBoostClassifier::new()
            .with_n_estimators(6)
            .unwrap()
            .fit(&dataset)
            .unwrap_err(),
        MlError::WeakLearnerNoBetterThanRandom
    );
}

#[test]
fn probabilities_are_normalized() {
    let model = AdaBoostClassifier::new()
        .with_n_estimators(6)
        .unwrap()
        .fit(&noisy_dataset())
        .unwrap();

    let probabilities = model
        .predict_probabilities(array![[-4.0], [-0.25], [0.25], [4.0]].view())
        .unwrap();

    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-12);
        assert!(row.iter().all(|&value| (0.0..=1.0).contains(&value)));
    }
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = AdaBoostClassifier::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn rejects_target_collections_without_exactly_two_classes() {
    let single_class = Dataset::new(array![[0.0], [1.0]], array!["no", "no"]).unwrap();
    assert_eq!(
        AdaBoostClassifier::new().fit(&single_class).unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 1 }
    );

    let three_classes = Dataset::new(array![[0.0], [1.0], [2.0]], array!["a", "b", "c"]).unwrap();
    assert_eq!(
        AdaBoostClassifier::new().fit(&three_classes).unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 3 }
    );
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = AdaBoostClassifier::default();
    assert_eq!(default.n_estimators(), 50);
    assert_abs_diff_eq!(default.learning_rate(), 1.0);
    assert_eq!(default.max_depth(), Some(1));
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);

    assert_eq!(
        AdaBoostClassifier::new().with_n_estimators(0).unwrap_err(),
        MlError::InvalidEstimatorCount(0)
    );
    assert_eq!(
        AdaBoostClassifier::new()
            .with_learning_rate(-1.0)
            .unwrap_err(),
        MlError::InvalidLearningRate(-1.0)
    );
    assert_eq!(
        AdaBoostClassifier::new()
            .with_min_samples_split(1)
            .unwrap_err(),
        MlError::InvalidMinSamplesSplit(1)
    );
    assert_eq!(
        AdaBoostClassifier::new()
            .with_min_samples_leaf(0)
            .unwrap_err(),
        MlError::InvalidMinSamplesLeaf(0)
    );
}

#[test]
fn validates_prediction_features() {
    let model = AdaBoostClassifier::new()
        .with_n_estimators(5)
        .unwrap()
        .fit(&separable_dataset())
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
fn composes_with_stratified_cross_validation() {
    let data = noisy_dataset();
    let labels = data.targets().to_vec();
    let folds = StratifiedKFold::new(5).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &AdaBoostClassifier::new().with_n_estimators(6).unwrap(),
        &data,
        &folds,
        accuracy_score,
    )
    .unwrap();

    assert_eq!(scores.scores().len(), 5);
    assert!(
        scores
            .scores()
            .iter()
            .all(|&score| (0.0..=1.0).contains(&score))
    );
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = AdaBoostClassifier::new().with_n_estimators(6).unwrap();
    let model = estimator.fit(&noisy_dataset()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: AdaBoostClassifier = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedAdaBoostClassifier<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[-0.25], [0.25]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        restored_model.predict(records.view()).unwrap()
    );
}
