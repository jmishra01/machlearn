//! Integration tests for linear discriminant analysis.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, LinearDiscriminantAnalysis, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn three_class_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![
            [-3.0, 0.0],
            [-2.0, 0.5],
            [-2.0, -0.5],
            [3.0, 0.0],
            [2.0, 0.5],
            [2.0, -0.5],
            [0.0, 3.0],
            [0.5, 2.0],
            [-0.5, 2.0],
        ],
        array![
            "amber", "amber", "amber", "blue", "blue", "blue", "cyan", "cyan", "cyan",
        ],
    )
    .unwrap()
}

#[test]
fn matches_a_reference_solution() {
    // Reference values confirmed against
    // `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr")`
    // fitted on the same data.
    let model = LinearDiscriminantAnalysis::new()
        .fit(&three_class_dataset())
        .unwrap();

    assert_eq!(model.classes(), &["amber", "blue", "cyan"]);
    assert_abs_diff_eq!(
        model.coefficients()[[0, 0]],
        -11.454_545_454_545_455,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(model.coefficients()[[0, 1]], 0.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(
        model.coefficients()[[1, 0]],
        11.454_545_454_545_455,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(model.coefficients()[[2, 1]], 12.6, epsilon = 1.0e-9);
    assert_abs_diff_eq!(
        model.intercepts()[0],
        -14.462_248_652_304_474,
        epsilon = 1.0e-8
    );
    assert_abs_diff_eq!(
        model.intercepts()[2],
        -15.798_612_288_668_112,
        epsilon = 1.0e-8
    );

    let query = array![[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.999_999_999_999_9, epsilon = 1.0e-9);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.999_999_999_999_9, epsilon = 1.0e-9);
    assert_abs_diff_eq!(probabilities[[2, 2]], 0.999_999_999_999_8, epsilon = 1.0e-9);
    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-9);
    }
    assert_eq!(
        model.predict(query.view()).unwrap(),
        array!["amber", "blue", "cyan"]
    );
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = LinearDiscriminantAnalysis::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn works_with_two_classes() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )
    .unwrap();
    let model = LinearDiscriminantAnalysis::new().fit(&dataset).unwrap();

    assert_eq!(model.n_classes(), 2);
    let predictions = model.predict(array![[-0.5], [0.5]].view()).unwrap();
    assert_eq!(predictions, array!["no", "yes"]);
}

#[test]
fn rejects_a_rank_deficient_covariance() {
    // A single feature column with every value identical within each class
    // (and a second column that is an exact multiple of the first) leaves
    // the pooled covariance singular.
    let dataset = Dataset::new(
        array![[1.0, 2.0], [1.0, 2.0], [2.0, 4.0], [2.0, 4.0]],
        array!["a", "a", "b", "b"],
    )
    .unwrap();

    let result = LinearDiscriminantAnalysis::new().fit(&dataset);

    assert!(matches!(result, Err(MlError::RankDeficientDesign { .. })));
}

#[test]
fn validates_prediction_features() {
    let model = LinearDiscriminantAnalysis::new()
        .fit(&three_class_dataset())
        .unwrap();

    assert_eq!(
        model.predict(array![[1.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        model
            .predict_probabilities(array![[f64::NAN, 0.0]].view())
            .unwrap_err(),
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
        &LinearDiscriminantAnalysis::new(),
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
    let estimator = LinearDiscriminantAnalysis::new();
    let dataset = three_class_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: LinearDiscriminantAnalysis =
        serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedLinearDiscriminantAnalysis<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]];
    let predictions = model.predict(records.view()).unwrap();
    let restored_predictions = restored_model.predict(records.view()).unwrap();
    for (left, right) in predictions.iter().zip(restored_predictions.iter()) {
        assert_eq!(left, right);
    }
}
