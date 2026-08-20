//! Integration tests for the CART-style decision-tree regressor.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, DecisionTreeRegressor, KFold, MlError, cross_validate, mean_squared_error,
};
use ndarray::array;

#[test]
fn matches_a_reference_split_boundary() {
    // Reference split (`feature_0 <= 2.50`) and predictions confirmed against
    // `sklearn.tree.DecisionTreeRegressor` fitted on the same data.
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
    )
    .unwrap();
    let model = DecisionTreeRegressor::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[0.5], [3.5]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(predictions[1], 10.0, epsilon = 1.0e-12);
}

#[test]
fn feature_importances_matches_a_reference_solution() {
    // Reference importances ([0.80327869, 0.19672131]) confirmed against
    // `sklearn.tree.DecisionTreeRegressor` fitted on the same data.
    let dataset = Dataset::new(
        array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        array![0.0, 0.0, 5.0, 5.0, 9.0, 9.0],
    )
    .unwrap();
    let model = DecisionTreeRegressor::new().fit(&dataset).unwrap();

    let importances = model.feature_importances();

    assert_abs_diff_eq!(importances[0], 0.803_278_69, epsilon = 1.0e-6);
    assert_abs_diff_eq!(importances[1], 0.196_721_31, epsilon = 1.0e-6);
    assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1.0e-12);
}

#[test]
fn feature_importances_are_zero_for_an_unsplit_tree() {
    let dataset = Dataset::new(
        array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        array![0.0, 1.0, 2.0],
    )
    .unwrap();
    let model = DecisionTreeRegressor::new()
        .with_max_depth(Some(0))
        .fit(&dataset)
        .unwrap();

    let importances = model.feature_importances();

    assert_abs_diff_eq!(importances[0], 0.0);
    assert_abs_diff_eq!(importances[1], 0.0);
}

#[test]
fn respects_a_configured_max_depth() {
    // Reference split (`feature_0 <= 1.50`) confirmed against
    // `sklearn.tree.DecisionTreeRegressor(max_depth=1)` fitted on the same data.
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0]],
        array![0.0, 0.0, 10.0, 10.0],
    )
    .unwrap();
    let model = DecisionTreeRegressor::new()
        .with_max_depth(Some(1))
        .fit(&dataset)
        .unwrap();

    let predictions = model.predict(array![[0.5], [2.5]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(predictions[1], 10.0, epsilon = 1.0e-12);
}

#[test]
fn max_depth_zero_predicts_the_overall_mean() {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0]],
        array![0.0, 0.0, 10.0, 10.0],
    )
    .unwrap();
    let model = DecisionTreeRegressor::new()
        .with_max_depth(Some(0))
        .fit(&dataset)
        .unwrap();

    let predictions = model.predict(array![[0.5], [2.5]].view()).unwrap();

    assert_abs_diff_eq!(predictions[0], 5.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(predictions[1], 5.0, epsilon = 1.0e-12);
}

#[test]
fn min_samples_leaf_prevents_an_unbalanced_split() {
    let dataset = Dataset::new(array![[0.0], [1.0], [2.0]], array![0.0, 0.0, 9.0]).unwrap();
    let model = DecisionTreeRegressor::new()
        .with_min_samples_leaf(2)
        .unwrap()
        .fit(&dataset)
        .unwrap();

    let predictions = model.predict(array![[0.0], [1.0], [2.0]].view()).unwrap();

    // No split leaves both sides with at least two samples, so the tree stays
    // a single leaf predicting the overall mean.
    for &prediction in &predictions {
        assert_abs_diff_eq!(prediction, 3.0, epsilon = 1.0e-12);
    }
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = DecisionTreeRegressor::default();
    assert_eq!(default.max_depth(), None);
    assert_eq!(default.min_samples_split(), 2);
    assert_eq!(default.min_samples_leaf(), 1);

    let estimator = DecisionTreeRegressor::new()
        .with_max_depth(Some(3))
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap();
    assert_eq!(estimator.max_depth(), Some(3));
    assert_eq!(estimator.min_samples_split(), 4);
    assert_eq!(estimator.min_samples_leaf(), 2);

    assert_eq!(
        DecisionTreeRegressor::new()
            .with_min_samples_split(1)
            .unwrap_err(),
        MlError::InvalidMinSamplesSplit(1)
    );
    assert_eq!(
        DecisionTreeRegressor::new()
            .with_min_samples_leaf(0)
            .unwrap_err(),
        MlError::InvalidMinSamplesLeaf(0)
    );

    let non_finite_targets = Dataset::new(array![[0.0], [1.0]], array![0.0, f64::NAN]).unwrap();
    assert_eq!(
        DecisionTreeRegressor::new()
            .fit(&non_finite_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn validates_prediction_features() {
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![-1.0, 1.0]).unwrap();
    let model = DecisionTreeRegressor::new().fit(&dataset).unwrap();

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
        &DecisionTreeRegressor::new(),
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
    let estimator = DecisionTreeRegressor::new().with_max_depth(Some(2));
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![-1.0, 1.0]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: DecisionTreeRegressor = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedDecisionTreeRegressor =
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
