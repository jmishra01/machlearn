//! Integration tests for dataset construction and validation.

use machlearn::{Dataset, MlError};
use ndarray::{Array1, Array2, array};

#[test]
fn creates_a_valid_dataset() {
    let dataset = Dataset::new(array![[1.0, 2.0], [3.0, 4.0]], array![0_u8, 1]).unwrap();

    assert_eq!(dataset.shape(), (2, 2));
    assert_eq!(dataset.n_samples(), 2);
    assert_eq!(dataset.n_features(), 2);
    assert_eq!(dataset.targets(), array![0_u8, 1].view());
}

#[test]
fn rejects_empty_samples() {
    let error = Dataset::<f64>::new(Array2::zeros((0, 2)), Array1::zeros(0)).unwrap_err();
    assert_eq!(error, MlError::EmptySamples);
}

#[test]
fn rejects_empty_features() {
    let error = Dataset::new(Array2::zeros((2, 0)), array![1.0, 2.0]).unwrap_err();
    assert_eq!(error, MlError::EmptyFeatures);
}

#[test]
fn rejects_mismatched_targets() {
    let error = Dataset::new(array![[1.0], [2.0]], array![1.0]).unwrap_err();
    assert_eq!(
        error,
        MlError::MismatchedSampleCount {
            feature_rows: 2,
            target_count: 1,
        }
    );
}

#[test]
fn rejects_non_finite_features() {
    let error = Dataset::new(array![[1.0, f64::NAN]], array![1.0]).unwrap_err();
    assert_eq!(error, MlError::NonFiniteFeature { row: 0, column: 1 });
}
