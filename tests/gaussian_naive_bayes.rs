//! Integration tests for Gaussian Naive Bayes classification.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, GaussianNaiveBayes, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn symmetric_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )
    .unwrap()
}

#[test]
fn matches_a_symmetric_reference_solution() {
    // Reference values from `sklearn.naive_bayes.GaussianNB` fitted on the
    // same data.
    let model = GaussianNaiveBayes::new().fit(&symmetric_dataset()).unwrap();

    assert_eq!(model.classes(), &["no", "yes"]);
    assert_abs_diff_eq!(
        model.means()[[0, 0]],
        -1.166_666_666_666_666_7,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(
        model.means()[[1, 0]],
        1.166_666_666_666_666_7,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(
        model.variances()[[0, 0]],
        0.388_888_888_888_888_9,
        epsilon = 1.0e-8
    );
    assert_abs_diff_eq!(
        model.variances()[[1, 0]],
        0.388_888_888_888_888_9,
        epsilon = 1.0e-8
    );
    assert_abs_diff_eq!(model.class_priors()[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.class_priors()[1], 0.5, epsilon = 1.0e-12);

    let records = array![[-1.5], [0.0], [1.5]];
    let probabilities = model.predict_probabilities(records.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.999_876_605, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.5, epsilon = 1.0e-9);
    assert_abs_diff_eq!(probabilities[[2, 1]], 0.999_876_605, epsilon = 1.0e-6);
    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-12);
    }

    let predictions = model.predict(records.view()).unwrap();
    assert_eq!(predictions, array!["no", "no", "yes"]);
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = GaussianNaiveBayes::new().fit(&dataset).unwrap();

    let predictions = model.predict(array![[-4.0], [4.0]].view()).unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn exposes_configuration_and_validates_variance_smoothing() {
    let default = GaussianNaiveBayes::default();
    assert_abs_diff_eq!(default.var_smoothing(), 1.0e-9);

    let estimator = GaussianNaiveBayes::new().with_var_smoothing(0.1).unwrap();
    assert_abs_diff_eq!(estimator.var_smoothing(), 0.1);

    assert_eq!(
        GaussianNaiveBayes::new()
            .with_var_smoothing(-1.0)
            .unwrap_err(),
        MlError::InvalidVarianceSmoothing(-1.0)
    );
    assert!(matches!(
        GaussianNaiveBayes::new().with_var_smoothing(f64::NAN),
        Err(MlError::InvalidVarianceSmoothing(value)) if value.is_nan()
    ));
    assert_eq!(
        GaussianNaiveBayes::new()
            .with_var_smoothing(f64::INFINITY)
            .unwrap_err(),
        MlError::InvalidVarianceSmoothing(f64::INFINITY)
    );

    let model = GaussianNaiveBayes::new().fit(&symmetric_dataset()).unwrap();
    assert_eq!(model.n_classes(), 2);
    assert_eq!(model.n_features(), 1);
}

#[test]
fn validates_prediction_features_and_scores() {
    let model = GaussianNaiveBayes::new().fit(&symmetric_dataset()).unwrap();

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
        cross_validate(&GaussianNaiveBayes::new(), &dataset, &folds, accuracy_score).unwrap();

    assert_eq!(scores.scores(), &[1.0, 1.0, 1.0]);
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = GaussianNaiveBayes::new().with_var_smoothing(0.01).unwrap();
    let dataset = symmetric_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: GaussianNaiveBayes = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedGaussianNaiveBayes<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    assert_abs_diff_eq!(model.means()[[0, 0]], restored_model.means()[[0, 0]]);
    assert_abs_diff_eq!(
        model.variances()[[0, 0]],
        restored_model.variances()[[0, 0]]
    );
}
