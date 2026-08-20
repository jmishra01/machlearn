//! Integration tests for the bagged random-forest classifier.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, MaxFeatures, MlError, RandomForestClassifier, StratifiedKFold, accuracy_score,
    cross_validate,
};
use ndarray::{Array1, Array2, array};

/// A large, well-separated dataset (15 samples per class). With this many
/// samples per class, a bootstrap resample collapsing onto a single class
/// has negligible probability (`0.5^30`), so per-tree bagging noise does not
/// perturb predictions for interior query points.
fn separated_dataset() -> Dataset<&'static str> {
    let low = -10..5;
    let high = 5..20;
    let records: Vec<f64> = low.clone().chain(high.clone()).map(f64::from).collect();
    let targets: Vec<&'static str> = low.map(|_| "no").chain(high.map(|_| "yes")).collect();
    let n_samples = records.len();
    Dataset::new(
        Array2::from_shape_vec((n_samples, 1), records).unwrap(),
        Array1::from_vec(targets),
    )
    .unwrap()
}

#[test]
fn predicts_the_label_of_well_separated_clusters() {
    let model = RandomForestClassifier::new()
        .fit(&separated_dataset())
        .unwrap();

    let predictions = model
        .predict(array![[-8.0], [18.0], [100.0]].view())
        .unwrap();

    assert_eq!(predictions, array!["no", "yes", "yes"]);
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
    let targets: Vec<&'static str> = low.map(|_| "no").chain(high.map(|_| "yes")).collect();
    let n_samples = targets.len();
    let dataset = Dataset::new(
        Array2::from_shape_vec((n_samples, 2), flat).unwrap(),
        Array1::from_vec(targets),
    )
    .unwrap();

    let model = RandomForestClassifier::new()
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
    let estimator = RandomForestClassifier::new()
        .with_n_estimators(20)
        .unwrap()
        .with_seed(7);
    let first = estimator.fit(&separated_dataset()).unwrap();
    let second = estimator.fit(&separated_dataset()).unwrap();

    assert_eq!(first, second);
}

#[test]
fn probabilities_are_normalized_and_average_across_trees() {
    let model = RandomForestClassifier::new()
        .with_n_estimators(20)
        .unwrap()
        .fit(&separated_dataset())
        .unwrap();

    let probabilities = model
        .predict_probabilities(array![[-8.0], [18.0]].view())
        .unwrap();

    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-12);
        assert!(row.iter().all(|&value| (0.0..=1.0).contains(&value)));
    }
    assert!(probabilities[[0, 0]] > probabilities[[0, 1]]);
    assert!(probabilities[[1, 1]] > probabilities[[1, 0]]);
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = RandomForestClassifier::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = RandomForestClassifier::default();
    assert_eq!(default.n_estimators(), 100);
    assert_eq!(default.max_depth(), None);
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);
    assert_eq!(default.max_features(), MaxFeatures::Sqrt);
    assert_eq!(default.seed(), 42);

    let estimator = RandomForestClassifier::new()
        .with_n_estimators(5)
        .unwrap()
        .with_max_depth(Some(3))
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap()
        .with_max_features(MaxFeatures::Fixed(1))
        .unwrap()
        .with_seed(99);
    assert_eq!(estimator.n_estimators(), 5);
    assert_eq!(estimator.max_depth(), Some(3));
    assert_eq!(estimator.min_samples_split(), 4);
    assert_eq!(estimator.min_samples_leaf(), 2);
    assert_eq!(estimator.max_features(), MaxFeatures::Fixed(1));
    assert_eq!(estimator.seed(), 99);

    assert_eq!(
        RandomForestClassifier::new()
            .with_n_estimators(0)
            .unwrap_err(),
        MlError::InvalidEstimatorCount(0)
    );
    assert_eq!(
        RandomForestClassifier::new()
            .with_max_features(MaxFeatures::Fixed(0))
            .unwrap_err(),
        MlError::InvalidMaxFeatures(0)
    );
    assert_eq!(
        RandomForestClassifier::new()
            .with_min_samples_split(1)
            .unwrap_err(),
        MlError::InvalidMinSamplesSplit(1)
    );
}

#[test]
fn validates_prediction_features() {
    let model = RandomForestClassifier::new()
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
fn composes_with_stratified_cross_validation() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let labels = dataset.targets().to_vec();
    let folds = StratifiedKFold::new(3).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &RandomForestClassifier::new().with_n_estimators(10).unwrap(),
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
    let estimator = RandomForestClassifier::new()
        .with_n_estimators(5)
        .unwrap()
        .with_seed(3);
    let dataset = separated_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: RandomForestClassifier = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedRandomForestClassifier<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[-2.5], [2.5]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        restored_model.predict(records.view()).unwrap()
    );
}
