//! Integration tests for transformer pipelines.

use approx::assert_abs_diff_eq;
use machlearn::{
    FittedTransformer, MinMaxScaler, Pipeline, Result, SimpleImputer, StandardScaler,
    TransformerEstimator,
};
use ndarray::{Array2, ArrayView2, array};

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
