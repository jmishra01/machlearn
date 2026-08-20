//! Integration tests for one-vs-rest multiclass logistic regression.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, MlError, MulticlassLogisticRegression, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn three_class_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![
            [-3.0, 0.0],
            [-2.0, -0.5],
            [-2.0, 0.5],
            [3.0, 0.0],
            [2.0, -0.5],
            [2.0, 0.5],
            [0.0, 3.0],
            [-0.5, 2.0],
            [0.5, 2.0],
        ],
        array![
            "amber", "amber", "amber", "blue", "blue", "blue", "cyan", "cyan", "cyan"
        ],
    )
    .unwrap()
}

#[test]
fn matches_a_reference_solution() {
    // Reference values confirmed against `sklearn.multiclass.OneVsRestClassifier`
    // wrapping `sklearn.linear_model.LogisticRegression(C=1.0, tol=1e-12,
    // max_iter=10000)` fitted on the same data. `C = 1 / alpha` at `alpha =
    // 1.0`, matching this crate's default regularization strength.
    let dataset = Dataset::new(
        array![
            [-3.0, 0.0],
            [-2.0, 0.5],
            [3.0, 0.0],
            [2.0, 0.5],
            [0.0, 3.0],
            [0.5, 2.0],
        ],
        array!["amber", "amber", "blue", "blue", "cyan", "cyan"],
    )
    .unwrap();

    let model = MulticlassLogisticRegression::new().fit(&dataset).unwrap();

    assert_abs_diff_eq!(
        model.coefficients()[[0, 0]],
        -0.985_550_636_950_342_2,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.coefficients()[[0, 1]],
        -0.491_375_503_124_198_5,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.coefficients()[[1, 0]],
        0.950_683_190_156_297_9,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.coefficients()[[1, 1]],
        -0.566_198_148_506_178_1,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.coefficients()[[2, 0]],
        0.052_038_625_470_692_5,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.coefficients()[[2, 1]],
        1.192_090_913_243_086_1,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.intercepts()[0],
        -0.555_433_929_100_422_6,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.intercepts()[1],
        -0.666_396_380_425_123_3,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        model.intercepts()[2],
        -2.072_723_727_489_041_8,
        epsilon = 1.0e-6
    );

    let query = array![[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(
        probabilities[[0, 0]],
        0.857_250_062_332_900_5,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        probabilities[[0, 1]],
        0.044_805_599_130_777_5,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        probabilities[[0, 2]],
        0.097_944_338_536_322,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        probabilities[[1, 1]],
        0.831_251_423_564_473_8,
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        probabilities[[2, 2]],
        0.736_663_401_548_867,
        epsilon = 1.0e-6
    );
    assert_eq!(
        model.predict(query.view()).unwrap(),
        array!["amber", "blue", "cyan"]
    );
}

#[test]
fn learns_sorted_classes_and_parameter_shapes() {
    let dataset = three_class_dataset();

    let model = MulticlassLogisticRegression::new().fit(&dataset).unwrap();

    assert_eq!(model.classes(), &["amber", "blue", "cyan"]);
    assert_eq!(model.n_classes(), 3);
    assert_eq!(model.n_features(), 2);
    assert_eq!(model.coefficients().dim(), (3, 2));
    assert_eq!(model.intercepts().len(), 3);
    assert_abs_diff_eq!(model.alpha(), 1.0);
}

#[test]
fn predicts_original_labels_for_separated_clusters() {
    let dataset = three_class_dataset();
    let model = MulticlassLogisticRegression::new().fit(&dataset).unwrap();

    let predictions = model
        .predict(array![[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]].view())
        .unwrap();

    assert_eq!(predictions, array!["amber", "blue", "cyan"]);
}

#[test]
fn produces_normalized_probabilities_in_class_order() {
    let dataset = three_class_dataset();
    let model = MulticlassLogisticRegression::new().fit(&dataset).unwrap();
    let records = array![[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]];

    let scores = model.decision_function(records.view()).unwrap();
    let probabilities = model.predict_probabilities(records.view()).unwrap();

    assert_eq!(scores.dim(), (3, 3));
    assert_eq!(probabilities.dim(), (3, 3));
    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-12);
        assert!(row.iter().all(|value| value.is_finite()));
        assert!(row.iter().all(|value| (0.0..=1.0).contains(value)));
    }
    assert!(probabilities[[0, 0]] > probabilities[[0, 1]]);
    assert!(probabilities[[1, 1]] > probabilities[[1, 2]]);
    assert!(probabilities[[2, 2]] > probabilities[[2, 0]]);
}

#[test]
fn exposes_regularization_and_intercept_configuration() {
    let estimator = MulticlassLogisticRegression::new()
        .with_regularization(0.5)
        .unwrap()
        .with_intercept(false);
    let model = estimator.fit(&three_class_dataset()).unwrap();

    assert_abs_diff_eq!(estimator.alpha(), 0.5);
    assert!(!estimator.fit_intercept());
    assert!(model.intercepts().iter().all(|value| *value == 0.0));
    assert_abs_diff_eq!(MulticlassLogisticRegression::default().alpha(), 1.0);
    assert!(MulticlassLogisticRegression::default().fit_intercept());

    assert_eq!(
        MulticlassLogisticRegression::new()
            .with_regularization(-1.0)
            .unwrap_err(),
        MlError::InvalidRegularization(-1.0)
    );
    assert!(matches!(
        MulticlassLogisticRegression::new().with_regularization(f64::NAN),
        Err(MlError::InvalidRegularization(value)) if value.is_nan()
    ));
}

#[test]
fn reports_per_class_convergence_and_honors_configured_stopping_criteria() {
    let dataset = three_class_dataset();
    let estimator = MulticlassLogisticRegression::new()
        .with_max_iterations(50)
        .unwrap()
        .with_tolerance(1.0e-8)
        .unwrap();

    assert_eq!(estimator.max_iterations(), 50);
    assert_abs_diff_eq!(estimator.tolerance(), 1.0e-8);

    let model = estimator.fit(&dataset).unwrap();
    let reports = model.convergence_reports();

    assert_eq!(reports.len(), model.n_classes());
    for report in reports {
        assert!(report.iterations() >= 1);
        assert!(report.iterations() <= 50);
        assert_abs_diff_eq!(report.tolerance(), 1.0e-8);
    }

    assert_eq!(
        MulticlassLogisticRegression::new()
            .with_max_iterations(0)
            .unwrap_err(),
        MlError::InvalidMaxIterations(0)
    );
    assert_eq!(
        MulticlassLogisticRegression::new()
            .with_tolerance(0.0)
            .unwrap_err(),
        MlError::InvalidTolerance(0.0)
    );

    let unreachable_tolerance = MulticlassLogisticRegression::new()
        .with_max_iterations(1)
        .unwrap();
    assert_eq!(
        unreachable_tolerance.fit(&dataset).unwrap_err(),
        MlError::OptimizationDidNotConverge { iterations: 1 }
    );
}

#[test]
fn requires_at_least_three_classes() {
    let one_class = Dataset::new(array![[0.0], [1.0]], array!["same", "same"]).unwrap();
    assert_eq!(
        MulticlassLogisticRegression::new()
            .fit(&one_class)
            .unwrap_err(),
        MlError::ExpectedMulticlassTargets { class_count: 1 }
    );

    let two_classes = Dataset::new(array![[-1.0], [1.0]], array![0_u8, 1]).unwrap();
    assert_eq!(
        MulticlassLogisticRegression::new()
            .fit(&two_classes)
            .unwrap_err(),
        MlError::ExpectedMulticlassTargets { class_count: 2 }
    );
}

#[test]
fn validates_prediction_features_and_scores() {
    let model = MulticlassLogisticRegression::new()
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
    let dataset = three_class_dataset();
    let labels = dataset.targets().to_vec();
    let folds = StratifiedKFold::new(3).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &MulticlassLogisticRegression::new(),
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
    let estimator = MulticlassLogisticRegression::new()
        .with_regularization(0.5)
        .unwrap();
    let dataset = Dataset::new(
        array![[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]],
        array![
            "amber".to_owned(),
            "amber".to_owned(),
            "blue".to_owned(),
            "blue".to_owned(),
            "cyan".to_owned(),
            "cyan".to_owned(),
        ],
    )
    .unwrap();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: MulticlassLogisticRegression =
        serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedMulticlassLogisticRegression<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model, restored_model);
}
