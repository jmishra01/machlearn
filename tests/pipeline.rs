//! Integration tests for transformer pipelines.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, FittedTransformer, KFold, LinearRegression, MinMaxScaler, Pipeline, Result,
    SimpleImputer, StandardScaler, TransformerEstimator, cross_validate, r2_score,
};
use ndarray::{Array1, Array2, ArrayView2, array};

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn fits_and_applies_steps_in_order() {
    let records = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    let pipeline = Pipeline::new()
        .then(StandardScaler::default())
        .then(MinMaxScaler::default());

    let fitted = pipeline.fit(records.view()).unwrap();
    let transformed = fitted.transform(records.view()).unwrap();

    assert_eq!(pipeline.len(), 2);
    assert_eq!(fitted.len(), 2);
    assert_matrix_close(&transformed, &array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);
}

#[test]
fn fitting_uses_only_the_training_matrix() {
    let training = array![[0.0], [10.0]];
    let unseen = array![[20.0]];
    let fitted = Pipeline::new()
        .then(StandardScaler::default())
        .then(MinMaxScaler::default())
        .fit(training.view())
        .unwrap();

    // The training values map to 0 and 1. A value beyond the training maximum
    // is intentionally not clipped and therefore maps to 2.
    assert_abs_diff_eq!(
        fitted.transform(unseen.view()).unwrap()[[0, 0]],
        2.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn an_empty_pipeline_is_an_identity_transform() {
    let records = array![[1.0, 2.0], [3.0, 4.0]];
    let pipeline = Pipeline::new();
    let fitted = pipeline.fit(records.view()).unwrap();

    assert!(pipeline.is_empty());
    assert!(fitted.is_empty());
    assert_matrix_close(&fitted.transform(records.view()).unwrap(), &records);
}

struct AddConstant(f64);

struct FittedAddConstant(f64);

impl TransformerEstimator for AddConstant {
    fn fit(&self, _records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(FittedAddConstant(self.0)))
    }
}

impl FittedTransformer for FittedAddConstant {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Ok(records.mapv(|value| value + self.0))
    }
}

#[test]
fn accepts_application_defined_transformers() {
    let records = array![[1.0], [2.0]];
    let fitted = Pipeline::new()
        .then(AddConstant(3.0))
        .fit(records.view())
        .unwrap();

    assert_matrix_close(
        &fitted.transform(records.view()).unwrap(),
        &array![[4.0], [5.0]],
    );
}

#[test]
fn imputer_can_clean_missing_values_for_later_steps() {
    let training = array![[1.0, f64::NAN], [3.0, 2.0], [5.0, 4.0]];
    let fitted = Pipeline::new()
        .then(SimpleImputer::mean())
        .then(StandardScaler::default())
        .fit(training.view())
        .unwrap();

    let transformed = fitted.transform(array![[f64::NAN, 3.0]].view()).unwrap();
    assert!(transformed.iter().all(|value| value.is_finite()));
    assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1.0e-12);
}

fn linear_dataset() -> Dataset<f64> {
    let records = array![
        [1.0, 5.0],
        [2.0, 3.0],
        [3.0, 8.0],
        [4.0, 1.0],
        [5.0, 6.0],
        [6.0, 2.0],
    ];
    // y = 2 * x0 + 3 * x1, an exact linear relationship so a scaler followed
    // by linear regression should recover it (through different, transformed
    // coefficients) without residual error.
    let targets: Array1<f64> = records
        .rows()
        .into_iter()
        .map(|row| 2.0 * row[0] + 3.0 * row[1])
        .collect();
    Dataset::new(records, targets).unwrap()
}

#[test]
fn pipeline_estimator_fits_and_predicts_end_to_end() {
    let dataset = linear_dataset();
    let pipeline = Pipeline::new()
        .then(StandardScaler::default())
        .with_estimator(LinearRegression::new());

    let fitted = pipeline.fit(&dataset).unwrap();

    let held_out = array![[7.0, 4.0], [0.5, 9.0]];
    let expected = array![2.0 * 7.0 + 3.0 * 4.0, 2.0 * 0.5 + 3.0 * 9.0];
    let predicted = fitted.predict(held_out.view()).unwrap();

    for (actual, expected) in predicted.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-9);
    }
}

#[test]
fn pipeline_estimator_composes_with_cross_validate() {
    let dataset = linear_dataset();
    let pipeline = Pipeline::new()
        .then(StandardScaler::default())
        .with_estimator(LinearRegression::new());
    let folds = KFold::new(3).unwrap().split(dataset.n_samples()).unwrap();

    let scores = cross_validate(&pipeline, &dataset, &folds, r2_score).unwrap();

    for &score in scores.scores() {
        assert_abs_diff_eq!(score, 1.0, epsilon = 1.0e-9);
    }
}
