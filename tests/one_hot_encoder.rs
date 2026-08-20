//! Integration tests for one-hot and dummy categorical encoding.

use machlearn::{MlError, OneHotEncoder};
use ndarray::{Array1, array};

#[test]
fn encodes_one_indicator_column_per_sorted_class() {
    // Reference column order and indicator values confirmed against
    // `sklearn.preprocessing.OneHotEncoder(sparse_output=False)` fitted on
    // the same data.
    let labels = array!["b", "a", "c", "a", "b"];
    let fitted = OneHotEncoder::new().fit(labels.view()).unwrap();
    let encoded = fitted.transform(labels.view()).unwrap();

    assert_eq!(fitted.classes(), &["a", "b", "c"]);
    assert_eq!(fitted.n_output_columns(), 3);
    assert_eq!(
        encoded,
        array![
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    );
}

#[test]
fn every_row_sums_to_one_without_drop_first() {
    let labels = array!["x", "y", "z", "x", "y"];
    let fitted = OneHotEncoder::new().fit(labels.view()).unwrap();
    let encoded = fitted.transform(labels.view()).unwrap();

    for row in encoded.rows() {
        assert!((row.sum() - 1.0).abs() < 1.0e-12);
    }
}

#[test]
fn drop_first_omits_the_first_sorted_class_as_an_all_zero_row() {
    // Reference confirmed against
    // `sklearn.preprocessing.OneHotEncoder(sparse_output=False,
    // drop="first")` fitted on the same data.
    let labels = array!["b", "a", "c", "a", "b"];
    let fitted = OneHotEncoder::new()
        .with_drop_first(true)
        .fit(labels.view())
        .unwrap();
    let encoded = fitted.transform(labels.view()).unwrap();

    assert_eq!(fitted.n_output_columns(), 2);
    assert_eq!(
        encoded,
        array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0],]
    );
}

#[test]
fn drop_first_on_a_single_class_produces_zero_columns() {
    let labels = array!["only", "only", "only"];
    let fitted = OneHotEncoder::new()
        .with_drop_first(true)
        .fit(labels.view())
        .unwrap();

    assert_eq!(fitted.n_output_columns(), 0);
    let encoded = fitted.transform(labels.view()).unwrap();
    assert_eq!(encoded.shape(), &[3, 0]);
}

#[test]
fn reports_the_first_unknown_label() {
    let fitted = OneHotEncoder::new()
        .fit(array!["cat", "dog"].view())
        .unwrap();
    let error = fitted
        .transform(array!["cat", "wolf", "dog"].view())
        .unwrap_err();

    assert_eq!(error, MlError::UnknownLabel { index: 1 });
}

#[test]
fn rejects_empty_label_collections() {
    let labels = Array1::<String>::from_vec(Vec::new());
    assert_eq!(
        OneHotEncoder::new().fit(labels.view()).unwrap_err(),
        MlError::EmptyTargets
    );
}

#[test]
fn supports_non_string_labels() {
    let labels = array![20_i16, 10, 20, 30];
    let fitted = OneHotEncoder::new().fit(labels.view()).unwrap();

    assert_eq!(fitted.classes(), &[10, 20, 30]);
    assert_eq!(
        fitted.transform(labels.view()).unwrap(),
        array![
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
}

#[test]
fn exposes_configuration() {
    let default = OneHotEncoder::default();
    assert!(!default.drop_first());

    let configured = OneHotEncoder::new().with_drop_first(true);
    assert!(configured.drop_first());
}

#[cfg(feature = "serde")]
#[test]
fn fitted_encoder_round_trips_through_serde() {
    let fitted = OneHotEncoder::new()
        .fit(array!["cat", "dog", "cat"].view())
        .unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let restored = serde_json::from_str(&json).unwrap();

    assert_eq!(fitted, restored);
}
