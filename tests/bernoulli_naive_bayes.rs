//! Integration tests for Bernoulli Naive Bayes classification.

use approx::assert_abs_diff_eq;
use machlearn::{
    BernoulliNaiveBayes, Dataset, MlError, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn presence_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        array!["a", "a", "b", "a", "b", "b"],
    )
    .unwrap()
}

#[test]
fn matches_sklearn_bernoulli_naive_bayes() {
    // Reference values confirmed against
    // `sklearn.naive_bayes.BernoulliNB(alpha=1.0, fit_prior=True)` fitted
    // on the same data.
    let model = BernoulliNaiveBayes::new().fit(&presence_dataset()).unwrap();

    assert_eq!(model.classes(), &["a", "b"]);
    assert_abs_diff_eq!(model.class_priors()[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.class_priors()[1], 0.5, epsilon = 1.0e-12);

    let query = array![[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.941_176_47, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.058_823_53, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.058_823_53, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.941_176_47, epsilon = 1.0e-6);

    let predictions = model.predict(query.view()).unwrap();
    assert_eq!(predictions, array!["a", "b"]);
}

#[test]
fn matches_sklearn_with_custom_alpha_and_uniform_prior() {
    // Reference values confirmed against
    // `sklearn.naive_bayes.BernoulliNB(alpha=0.5, fit_prior=False)` fitted
    // on the same data.
    let model = BernoulliNaiveBayes::new()
        .with_alpha(0.5)
        .unwrap()
        .with_fit_prior(false)
        .fit(&presence_dataset())
        .unwrap();

    let query = array![[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.98, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.02, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.02, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.98, epsilon = 1.0e-6);
}

#[test]
fn matches_sklearn_when_binarizing_count_features() {
    // Reference values confirmed against
    // `sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0)` fitted on
    // non-binary count data.
    let dataset = Dataset::new(
        array![
            [2.0, 0.0, 3.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 2.0, 1.0]
        ],
        array!["x", "y", "x", "y"],
    )
    .unwrap();
    let model = BernoulliNaiveBayes::new()
        .with_binarize(Some(0.0))
        .fit(&dataset)
        .unwrap();

    let query = array![[3.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.9, epsilon = 1.0e-9);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.1, epsilon = 1.0e-9);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.5, epsilon = 1.0e-9);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.5, epsilon = 1.0e-9);
}

#[test]
fn skipping_binarization_treats_features_as_already_binary() {
    let model_default = BernoulliNaiveBayes::new().fit(&presence_dataset()).unwrap();
    let model_no_binarize = BernoulliNaiveBayes::new()
        .with_binarize(None)
        .fit(&presence_dataset())
        .unwrap();

    // The training data is already 0/1, so skipping binarization changes
    // nothing: every feature value is already left unchanged by the
    // `value > 0.0` threshold used by default.
    assert_eq!(
        model_default.feature_log_prob(),
        model_no_binarize.feature_log_prob()
    );
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = BernoulliNaiveBayes::new().fit(&dataset).unwrap();

    let predictions = model
        .predict(array![[1.0, 0.0], [0.0, 1.0]].view())
        .unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn exposes_configuration_and_validates_alpha() {
    let default = BernoulliNaiveBayes::default();
    assert_abs_diff_eq!(default.alpha(), 1.0);
    assert!(default.fit_prior());
    assert_eq!(default.binarize(), Some(0.0));

    let estimator = BernoulliNaiveBayes::new()
        .with_alpha(0.1)
        .unwrap()
        .with_fit_prior(false)
        .with_binarize(None);
    assert_abs_diff_eq!(estimator.alpha(), 0.1);
    assert!(!estimator.fit_prior());
    assert_eq!(estimator.binarize(), None);

    assert_eq!(
        BernoulliNaiveBayes::new().with_alpha(-1.0).unwrap_err(),
        MlError::InvalidAlpha(-1.0)
    );
    assert!(matches!(
        BernoulliNaiveBayes::new().with_alpha(f64::NAN),
        Err(MlError::InvalidAlpha(value)) if value.is_nan()
    ));

    let model = BernoulliNaiveBayes::new().fit(&presence_dataset()).unwrap();
    assert_eq!(model.n_classes(), 2);
    assert_eq!(model.n_features(), 4);
}

#[test]
fn validates_prediction_features() {
    let model = BernoulliNaiveBayes::new().fit(&presence_dataset()).unwrap();

    assert_eq!(
        model.predict(array![[1.0, 2.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 4,
            actual: 2,
        }
    );
    assert_eq!(
        model
            .predict(array![[f64::NAN, 0.0, 0.0, 0.0]].view())
            .unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[test]
fn composes_with_stratified_cross_validation() {
    let dataset = Dataset::new(
        array![
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let labels = dataset.targets().to_vec();
    let folds = StratifiedKFold::new(3).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &BernoulliNaiveBayes::new(),
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
    let estimator = BernoulliNaiveBayes::new().with_alpha(0.5).unwrap();
    let dataset = presence_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: BernoulliNaiveBayes = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedBernoulliNaiveBayes<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[1.0, 1.0, 0.0, 0.0]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        restored_model.predict(records.view()).unwrap()
    );
}
