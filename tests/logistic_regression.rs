//! Integration tests for binary logistic regression.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, LogisticRegression, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

#[test]
fn matches_a_symmetric_regularized_reference_solution() {
    let dataset = Dataset::new(array![[-1.0], [1.0]], array!["no", "yes"]).unwrap();

    let model = LogisticRegression::new().fit(&dataset).unwrap();

    assert_abs_diff_eq!(model.intercept(), 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        model.coefficients()[0],
        0.674_831_614_342_399_6,
        epsilon = 1.0e-10
    );
    assert_abs_diff_eq!(model.alpha(), 1.0);
    assert_eq!(model.n_features(), 1);
}

#[test]
fn learns_sorted_classes_and_consistent_probabilities() {
    let dataset = Dataset::new(
        array![[-2.0], [-1.0], [1.0], [2.0]],
        array!["yes", "no", "yes", "yes"],
    )
    .unwrap();
    let model = LogisticRegression::new().fit(&dataset).unwrap();

    assert_eq!(model.classes(), &["no", "yes"]);
    assert_eq!(*model.negative_class(), "no");
    assert_eq!(*model.positive_class(), "yes");

    let records = array![[-1.0], [1.0]];
    let probabilities = model.predict_probabilities(records.view()).unwrap();
    let positive = model
        .predict_positive_probabilities(records.view())
        .unwrap();
    assert_eq!(probabilities.dim(), (2, 2));
    for row in 0..2 {
        assert_abs_diff_eq!(probabilities[[row, 0]] + probabilities[[row, 1]], 1.0);
        assert_abs_diff_eq!(probabilities[[row, 1]], positive[row]);
    }
    assert!(positive[0] < positive[1]);
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = LogisticRegression::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn supports_no_intercept_and_no_regularization() {
    let dataset = Dataset::new(
        array![[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        array![0_u8, 1, 0, 1, 1],
    )
    .unwrap();
    let estimator = LogisticRegression::new()
        .with_intercept(false)
        .with_regularization(0.0)
        .unwrap();

    let model = estimator.fit(&dataset).unwrap();

    assert!(!estimator.fit_intercept());
    assert_abs_diff_eq!(estimator.alpha(), 0.0);
    assert_abs_diff_eq!(model.intercept(), 0.0);
    assert!(model.coefficients()[0].is_finite());
}

#[test]
fn reports_convergence_and_honors_configured_stopping_criteria() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let estimator = LogisticRegression::new()
        .with_max_iterations(50)
        .unwrap()
        .with_tolerance(1.0e-8)
        .unwrap();

    assert_eq!(estimator.max_iterations(), 50);
    assert_abs_diff_eq!(estimator.tolerance(), 1.0e-8);

    let model = estimator.fit(&dataset).unwrap();

    assert!(model.convergence().iterations() >= 1);
    assert!(model.convergence().iterations() <= 50);
    assert_abs_diff_eq!(model.convergence().tolerance(), 1.0e-8);
    assert!(model.convergence().max_parameter_change() >= 0.0);

    assert_eq!(
        LogisticRegression::new()
            .with_max_iterations(0)
            .unwrap_err(),
        MlError::InvalidMaxIterations(0)
    );
    assert_eq!(
        LogisticRegression::new().with_tolerance(0.0).unwrap_err(),
        MlError::InvalidTolerance(0.0)
    );
    assert_eq!(
        LogisticRegression::new().with_tolerance(-1.0).unwrap_err(),
        MlError::InvalidTolerance(-1.0)
    );
    assert!(matches!(
        LogisticRegression::new().with_tolerance(f64::NAN),
        Err(MlError::InvalidTolerance(value)) if value.is_nan()
    ));

    let unreachable_tolerance = LogisticRegression::new().with_max_iterations(1).unwrap();
    assert_eq!(
        unreachable_tolerance.fit(&dataset).unwrap_err(),
        MlError::OptimizationDidNotConverge { iterations: 1 }
    );
}

#[test]
fn validates_regularization_and_binary_targets() {
    let default = LogisticRegression::default();
    assert_abs_diff_eq!(default.alpha(), 1.0);
    assert!(default.fit_intercept());

    assert_eq!(
        LogisticRegression::new()
            .with_regularization(-1.0)
            .unwrap_err(),
        MlError::InvalidRegularization(-1.0)
    );
    assert_eq!(
        LogisticRegression::new()
            .with_regularization(f64::INFINITY)
            .unwrap_err(),
        MlError::InvalidRegularization(f64::INFINITY)
    );
    assert!(matches!(
        LogisticRegression::new().with_regularization(f64::NAN),
        Err(MlError::InvalidRegularization(value)) if value.is_nan()
    ));

    let one_class = Dataset::new(array![[0.0], [1.0]], array!["same", "same"]).unwrap();
    assert_eq!(
        LogisticRegression::new().fit(&one_class).unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 1 }
    );
    let three_classes = Dataset::new(array![[0.0], [1.0], [2.0]], array![0_u8, 1, 2]).unwrap();
    assert_eq!(
        LogisticRegression::new().fit(&three_classes).unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 3 }
    );
}

#[test]
fn validates_prediction_features_and_scores() {
    let dataset = Dataset::new(array![[-1.0], [1.0]], array![0_u8, 1]).unwrap();
    let model = LogisticRegression::new().fit(&dataset).unwrap();

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

    let scores =
        cross_validate(&LogisticRegression::new(), &dataset, &folds, accuracy_score).unwrap();

    assert_eq!(scores.scores(), &[1.0, 1.0, 1.0]);
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = LogisticRegression::new()
        .with_regularization(0.5)
        .unwrap()
        .with_intercept(false);
    let dataset = Dataset::new(array![[-1.0], [1.0]], array!["no", "yes"]).unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: LogisticRegression = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedLogisticRegression<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    assert_abs_diff_eq!(model.intercept(), restored_model.intercept());
    assert_abs_diff_eq!(model.coefficients()[0], restored_model.coefficients()[0]);
}
