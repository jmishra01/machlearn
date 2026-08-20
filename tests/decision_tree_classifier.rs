//! Integration tests for the CART-style decision-tree classifier.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, DecisionTreeClassifier, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

#[test]
fn matches_a_reference_split_boundary() {
    // Reference split (`feature_0 <= 0.00`) and predictions confirmed against
    // `sklearn.tree.DecisionTreeClassifier` fitted on the same data.
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new().fit(&dataset).unwrap();

    let records = array![[-0.5], [0.5], [10.0]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        array!["no", "yes", "yes"]
    );

    let probabilities = model
        .predict_probabilities(array![[-0.5], [0.5]].view())
        .unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 1.0);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.0);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.0);
    assert_abs_diff_eq!(probabilities[[1, 1]], 1.0);
}

#[test]
fn feature_importances_matches_a_reference_solution() {
    // Reference importances ([0.5, 0.5]) confirmed against
    // `sklearn.tree.DecisionTreeClassifier` fitted on the same data: the
    // root splits on `feature_0` (separating "a" from "b"/"c"), and the
    // second split on `feature_1` separates "b" from "c".
    let dataset = Dataset::new(
        array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        array!["a", "a", "b", "b", "c", "c"],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new().fit(&dataset).unwrap();

    let importances = model.feature_importances();

    assert_abs_diff_eq!(importances[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(importances[1], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1.0e-12);
}

#[test]
fn feature_importances_are_zero_for_an_unsplit_tree() {
    let dataset = Dataset::new(
        array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        array!["a", "a", "b"],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new()
        .with_max_depth(Some(0))
        .fit(&dataset)
        .unwrap();

    let importances = model.feature_importances();

    assert_abs_diff_eq!(importances[0], 0.0);
    assert_abs_diff_eq!(importances[1], 0.0);
}

#[test]
fn ties_break_in_favor_of_the_first_feature() {
    // A symmetric XOR-like layout gives every feature the same weighted
    // impurity at the root; `sklearn` and this implementation both split on
    // `feature_0` first because it is evaluated first.
    let dataset = Dataset::new(
        array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        array!["a", "b", "b", "a"],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new().fit(&dataset).unwrap();

    let predictions = model.predict(dataset.records()).unwrap();

    assert_eq!(predictions, array!["a", "b", "b", "a"]);
}

#[test]
fn min_samples_leaf_prevents_an_unbalanced_split() {
    let dataset = Dataset::new(array![[0.0], [1.0], [2.0]], array!["a", "a", "b"]).unwrap();
    let model = DecisionTreeClassifier::new()
        .with_min_samples_leaf(2)
        .unwrap()
        .fit(&dataset)
        .unwrap();

    // Every candidate split leaves one side with a single sample, which
    // `min_samples_leaf = 2` forbids, so the tree stays a single majority-vote
    // leaf.
    let predictions = model.predict(array![[0.0], [1.0], [2.0]].view()).unwrap();
    assert_eq!(predictions, array!["a", "a", "a"]);
}

#[test]
fn max_depth_zero_predicts_the_overall_majority_class() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "no", "yes", "yes"],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new()
        .with_max_depth(Some(0))
        .fit(&dataset)
        .unwrap();

    let predictions = model.predict(array![[-10.0], [10.0]].view()).unwrap();

    assert_eq!(predictions, array!["no", "no"]);
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = DecisionTreeClassifier::default();
    assert_eq!(default.max_depth(), None);
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);

    let estimator = DecisionTreeClassifier::new()
        .with_max_depth(Some(3))
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap();
    assert_eq!(estimator.max_depth(), Some(3));
    assert_eq!(estimator.min_samples_split(), 4);
    assert_eq!(estimator.min_samples_leaf(), 2);

    assert_eq!(
        DecisionTreeClassifier::new()
            .with_min_samples_split(1)
            .unwrap_err(),
        MlError::InvalidMinSamplesSplit(1)
    );
    assert_eq!(
        DecisionTreeClassifier::new()
            .with_min_samples_leaf(0)
            .unwrap_err(),
        MlError::InvalidMinSamplesLeaf(0)
    );
}

#[test]
fn validates_prediction_features() {
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![0_u8, 1]).unwrap();
    let model = DecisionTreeClassifier::new().fit(&dataset).unwrap();

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
        &DecisionTreeClassifier::new(),
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
    let estimator = DecisionTreeClassifier::new()
        .with_max_depth(Some(2))
        .with_min_samples_split(2)
        .unwrap();
    let dataset = Dataset::new(array![[-1.0], [1.0]], array!["no", "yes"]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: DecisionTreeClassifier = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedDecisionTreeClassifier<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[-1.0], [1.0]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        restored_model.predict(records.view()).unwrap()
    );
}
