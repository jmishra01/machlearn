//! Integration tests for categorical label encoding.

use machlearn::{LabelEncoder, MlError};
use ndarray::{Array1, array};

#[test]
fn learns_sorted_classes_and_encodes_deterministically() {
    let labels = array!["dog", "cat", "fish", "dog"];
    let fitted = LabelEncoder::new().fit(labels.view()).unwrap();
    let encoded = fitted.transform(labels.view()).unwrap();

    assert_eq!(fitted.classes(), &["cat", "dog", "fish"]);
    assert_eq!(encoded, array![1, 0, 2, 1]);
}

#[test]
fn restores_original_labels() {
    let labels = array!["red", "blue", "green", "blue"];
    let fitted = LabelEncoder::new().fit(labels.view()).unwrap();
    let encoded = fitted.transform(labels.view()).unwrap();

    assert_eq!(fitted.inverse_transform(encoded.view()).unwrap(), labels);
}

#[test]
fn reports_the_first_unknown_label() {
    let fitted = LabelEncoder::new()
        .fit(array!["cat", "dog"].view())
        .unwrap();
    let error = fitted
        .transform(array!["cat", "wolf", "dog"].view())
        .unwrap_err();

    assert_eq!(error, MlError::UnknownLabel { index: 1 });
}

#[test]
fn reports_invalid_encoded_indices() {
    let fitted = LabelEncoder::new()
        .fit(array!["cat", "dog"].view())
        .unwrap();
    let error = fitted.inverse_transform(array![0, 2].view()).unwrap_err();

    assert_eq!(
        error,
        MlError::InvalidClassIndex {
            position: 1,
            class_index: 2,
            class_count: 2,
        }
    );
}

#[test]
fn rejects_empty_label_collections() {
    let labels = Array1::<String>::from_vec(Vec::new());
    assert_eq!(
        LabelEncoder::new().fit(labels.view()).unwrap_err(),
        MlError::EmptyTargets
    );
}

#[test]
fn supports_non_string_labels() {
    let labels = array![20_i16, 10, 20, 30];
    let fitted = LabelEncoder::new().fit(labels.view()).unwrap();

    assert_eq!(fitted.classes(), &[10, 20, 30]);
    assert_eq!(fitted.transform(labels.view()).unwrap(), array![1, 0, 1, 2]);
}

#[cfg(feature = "serde")]
#[test]
fn fitted_encoder_round_trips_through_serde() {
    let fitted = LabelEncoder::new()
        .fit(array!["cat", "dog", "cat"].view())
        .unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let restored = serde_json::from_str(&json).unwrap();

    assert_eq!(fitted, restored);
}
