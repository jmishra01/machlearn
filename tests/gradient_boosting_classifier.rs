//! Integration tests for the gradient-boosted decision-tree binary
//! classifier.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, GradientBoostingClassifier, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "no", "yes", "yes", "yes", "yes"],
    )
    .unwrap()
}

#[test]
fn matches_sklearn_gradient_boosting_classifier() {
    // Reference values confirmed against
    // `sklearn.ensemble.GradientBoostingClassifier(n_estimators=20,
    // learning_rate=0.1, max_depth=3, random_state=0)` fitted on the same
    // data. Matching sklearn's probabilities (not just its predicted
    // labels) requires a Newton-Raphson correction of each tree's leaf
    // values, since squared-error regression-tree leaf means are only the
    // correct update for squared-error loss.
    let model = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .unwrap()
        .fit(&dataset())
        .unwrap();

    let query = array![[-0.25], [0.25]];
    let predictions = model.predict(query.view()).unwrap();
    assert_eq!(predictions, array!["no", "yes"]);

    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.934_486_4, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.065_513_6, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.065_513_6, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.934_486_4, epsilon = 1.0e-6);
}

#[test]
fn probabilities_are_normalized() {
    let model = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .unwrap()
        .fit(&dataset())
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
    let model = GradientBoostingClassifier::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn rejects_target_collections_without_exactly_two_classes() {
    let single_class = Dataset::new(array![[0.0], [1.0]], array!["no", "no"]).unwrap();
    assert_eq!(
        GradientBoostingClassifier::new()
            .fit(&single_class)
            .unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 1 }
    );

    let three_classes = Dataset::new(array![[0.0], [1.0], [2.0]], array!["a", "b", "c"]).unwrap();
    assert_eq!(
        GradientBoostingClassifier::new()
            .fit(&three_classes)
            .unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 3 }
    );
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = GradientBoostingClassifier::default();
    assert_eq!(default.n_estimators(), 100);
    assert_abs_diff_eq!(default.learning_rate(), 0.1);
    assert_eq!(default.max_depth(), Some(3));
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);

    assert_eq!(
        GradientBoostingClassifier::new()
            .with_n_estimators(0)
            .unwrap_err(),
        MlError::InvalidEstimatorCount(0)
    );
    assert_eq!(
        GradientBoostingClassifier::new()
            .with_learning_rate(-1.0)
            .unwrap_err(),
        MlError::InvalidLearningRate(-1.0)
    );
}

#[test]
fn validates_prediction_features() {
    let model = GradientBoostingClassifier::new()
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
fn composes_with_stratified_cross_validation() {
    let data = dataset();
    let labels = data.targets().to_vec();
    let folds = StratifiedKFold::new(4).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .unwrap(),
        &data,
        &folds,
        accuracy_score,
    )
    .unwrap();

    assert_eq!(scores.scores().len(), 4);
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
    let estimator = GradientBoostingClassifier::new()
        .with_n_estimators(5)
        .unwrap();
    let model = estimator.fit(&dataset()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: GradientBoostingClassifier =
        serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedGradientBoostingClassifier<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[-2.5], [2.5]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        restored_model.predict(records.view()).unwrap()
    );
}
