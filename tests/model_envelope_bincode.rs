//! Integration tests for `bincode`-based model persistence.
#![cfg(feature = "bincode")]

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, ENVELOPE_VERSION, FittedLinearRegression, LinearRegression, MlError, ModelEnvelope,
};
use ndarray::array;

#[test]
fn round_trips_a_fitted_model_through_bincode_bytes() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = LinearRegression::new().fit(&dataset).unwrap();

    let bytes = ModelEnvelope::new(model.clone()).to_bincode().unwrap();
    let restored: ModelEnvelope<FittedLinearRegression> =
        ModelEnvelope::from_bincode(&bytes).unwrap();

    assert_eq!(restored.version(), ENVELOPE_VERSION);
    let restored_model = restored.into_model().unwrap();
    assert_abs_diff_eq!(restored_model.coefficients()[0], model.coefficients()[0]);
    assert_abs_diff_eq!(restored_model.intercept(), model.intercept());
}

#[test]
fn round_trips_a_fitted_model_through_a_bincode_file() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = LinearRegression::new().fit(&dataset).unwrap();

    let path = std::env::temp_dir().join(format!(
        "machlearn_test_{}_{}.bin",
        std::process::id(),
        "round_trips_a_fitted_model_through_a_bincode_file"
    ));
    ModelEnvelope::new(model.clone())
        .save_bincode_file(&path)
        .unwrap();

    let restored: FittedLinearRegression = ModelEnvelope::load_bincode_file(&path)
        .unwrap()
        .into_model()
        .unwrap();

    std::fs::remove_file(&path).unwrap();

    assert_abs_diff_eq!(restored.coefficients()[0], model.coefficients()[0]);
    assert_abs_diff_eq!(restored.intercept(), model.intercept());
}

#[test]
fn rejects_bytes_from_an_unsupported_envelope_version() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = LinearRegression::new().fit(&dataset).unwrap();

    // bincode encodes the envelope's `version: u32` as the first four bytes
    // (little-endian, no length prefix ahead of it), so it can be mutated
    // in place to simulate data written by an incompatible crate version.
    let mut bytes = ModelEnvelope::new(model).to_bincode().unwrap();
    bytes[0..4].copy_from_slice(&999_999u32.to_le_bytes());

    let envelope: ModelEnvelope<FittedLinearRegression> =
        ModelEnvelope::from_bincode(&bytes).unwrap();

    assert_eq!(
        envelope.into_model().unwrap_err(),
        MlError::UnsupportedEnvelopeVersion {
            found: 999_999,
            supported: ENVELOPE_VERSION,
        }
    );
}

#[test]
fn reports_a_serialization_error_for_garbage_bytes() {
    let error = ModelEnvelope::<FittedLinearRegression>::from_bincode(&[0xFF; 3]).unwrap_err();
    assert!(matches!(error, MlError::SerializationError(_)));
}
