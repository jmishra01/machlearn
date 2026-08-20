//! Integration tests for multinomial Naive Bayes classification.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, MlError, MultinomialNaiveBayes, StratifiedKFold, accuracy_score, cross_validate,
};
use ndarray::array;

fn count_dataset() -> Dataset<&'static str> {
    Dataset::new(
        array![
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 3.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [1.0, 3.0, 1.0],
        ],
        array!["a", "a", "b", "a", "b", "b"],
    )
    .unwrap()
}

#[test]
fn matches_sklearn_multinomial_naive_bayes() {
    // Reference values confirmed against
    // `sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True)` fitted
    // on the same data.
    let model = MultinomialNaiveBayes::new().fit(&count_dataset()).unwrap();

    assert_eq!(model.classes(), &["a", "b"]);
    assert_abs_diff_eq!(model.class_priors()[0], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.class_priors()[1], 0.5, epsilon = 1.0e-12);

    let query = array![[1.0, 1.0, 1.0], [0.0, 0.0, 5.0]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.525_093_52, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.474_906_48, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.003_082_73, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.996_917_27, epsilon = 1.0e-6);

    let predictions = model.predict(query.view()).unwrap();
    assert_eq!(predictions, array!["a", "b"]);
}

#[test]
fn matches_sklearn_with_custom_alpha_and_uniform_prior() {
    // Reference values confirmed against
    // `sklearn.naive_bayes.MultinomialNB(alpha=0.5, fit_prior=False)`
    // fitted on the same data.
    let model = MultinomialNaiveBayes::new()
        .with_alpha(0.5)
        .unwrap()
        .with_fit_prior(false)
        .fit(&count_dataset())
        .unwrap();

    let query = array![[1.0, 1.0, 1.0], [0.0, 0.0, 5.0]];
    let probabilities = model.predict_probabilities(query.view()).unwrap();
    assert_abs_diff_eq!(probabilities[[0, 0]], 0.527_889_26, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[0, 1]], 0.472_110_74, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 0]], 0.001_198_27, epsilon = 1.0e-6);
    assert_abs_diff_eq!(probabilities[[1, 1]], 0.998_801_73, epsilon = 1.0e-6);
}

#[test]
fn predicts_original_label_types() {
    let dataset = Dataset::new(
        array![
            [3.0, 0.0],
            [2.0, 0.0],
            [4.0, 1.0],
            [0.0, 3.0],
            [1.0, 2.0],
            [0.0, 4.0]
        ],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = MultinomialNaiveBayes::new().fit(&dataset).unwrap();

    let predictions = model
        .predict(array![[5.0, 0.0], [0.0, 5.0]].view())
        .unwrap();

    assert_eq!(predictions, array![0_u8, 1]);
}

#[test]
fn exposes_configuration_and_validates_alpha() {
    let default = MultinomialNaiveBayes::default();
    assert_abs_diff_eq!(default.alpha(), 1.0);
    assert!(default.fit_prior());

    let estimator = MultinomialNaiveBayes::new()
        .with_alpha(0.1)
        .unwrap()
        .with_fit_prior(false);
    assert_abs_diff_eq!(estimator.alpha(), 0.1);
    assert!(!estimator.fit_prior());

    assert_eq!(
        MultinomialNaiveBayes::new().with_alpha(-1.0).unwrap_err(),
        MlError::InvalidAlpha(-1.0)
    );
    assert!(matches!(
        MultinomialNaiveBayes::new().with_alpha(f64::NAN),
        Err(MlError::InvalidAlpha(value)) if value.is_nan()
    ));

    let model = MultinomialNaiveBayes::new().fit(&count_dataset()).unwrap();
    assert_eq!(model.n_classes(), 2);
    assert_eq!(model.n_features(), 3);
}

#[test]
fn rejects_negative_features_when_fitting_and_predicting() {
    let negative_dataset = Dataset::new(array![[1.0, -1.0], [2.0, 0.0]], array!["a", "b"]).unwrap();
    assert_eq!(
        MultinomialNaiveBayes::new()
            .fit(&negative_dataset)
            .unwrap_err(),
        MlError::NegativeFeature { row: 0, column: 1 }
    );

    let model = MultinomialNaiveBayes::new().fit(&count_dataset()).unwrap();
    assert_eq!(
        model.predict(array![[-1.0, 0.0, 0.0]].view()).unwrap_err(),
        MlError::NegativeFeature { row: 0, column: 0 }
    );
}

#[test]
fn validates_prediction_features() {
    let model = MultinomialNaiveBayes::new().fit(&count_dataset()).unwrap();

    assert_eq!(
        model.predict(array![[1.0, 2.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 3,
            actual: 2,
        }
    );
    assert_eq!(
        model
            .predict(array![[f64::NAN, 0.0, 0.0]].view())
            .unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[test]
fn composes_with_stratified_cross_validation() {
    let dataset = Dataset::new(
        array![
            [3.0, 0.0],
            [2.0, 0.0],
            [4.0, 1.0],
            [0.0, 3.0],
            [1.0, 2.0],
            [0.0, 4.0]
        ],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let labels = dataset.targets().to_vec();
    let folds = StratifiedKFold::new(3).unwrap().split(&labels).unwrap();

    let scores = cross_validate(
        &MultinomialNaiveBayes::new(),
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
    let estimator = MultinomialNaiveBayes::new().with_alpha(0.5).unwrap();
    let dataset = count_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: MultinomialNaiveBayes = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedMultinomialNaiveBayes<String> =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.classes(), restored_model.classes());
    let records = array![[1.0, 1.0, 1.0]];
    assert_eq!(
        model.predict(records.view()).unwrap(),
        restored_model.predict(records.view()).unwrap()
    );
}
