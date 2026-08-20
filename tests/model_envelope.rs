//! Integration tests for the versioned model-serialization envelope.
#![cfg(feature = "serde")]

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, ENVELOPE_VERSION, FittedLinearRegression, LinearRegression, MlError, ModelEnvelope,
};
use ndarray::array;

#[test]
fn round_trips_a_fitted_model_through_json() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = LinearRegression::new().fit(&dataset).unwrap();

    let envelope = ModelEnvelope::new(model.clone());
    assert_eq!(envelope.version(), ENVELOPE_VERSION);
    assert_eq!(envelope.model(), &model);

    let json = serde_json::to_string(&envelope).unwrap();
    let restored: ModelEnvelope<FittedLinearRegression> = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.version(), ENVELOPE_VERSION);
    let restored_model = restored.into_model().unwrap();
    assert_abs_diff_eq!(restored_model.coefficients()[0], model.coefficients()[0]);
    assert_abs_diff_eq!(restored_model.intercept(), model.intercept());
}

#[test]
fn rejects_an_unsupported_envelope_version() {
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0]).unwrap();
    let model = LinearRegression::new().fit(&dataset).unwrap();

    // Simulate data written by an incompatible future or past crate version
    // by editing the version field in the serialized JSON.
    let json = serde_json::to_string(&ModelEnvelope::new(model)).unwrap();
    let mutated = json.replacen(
        &format!("\"version\":{ENVELOPE_VERSION}"),
        "\"version\":999999",
        1,
    );
    let envelope: ModelEnvelope<FittedLinearRegression> = serde_json::from_str(&mutated).unwrap();

    assert_eq!(
        envelope.into_model().unwrap_err(),
        MlError::UnsupportedEnvelopeVersion {
            found: 999_999,
            supported: ENVELOPE_VERSION,
        }
    );
}
