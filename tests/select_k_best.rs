//! Integration tests for selecting the top-scoring features.

#[cfg(feature = "serde")]
use approx::assert_abs_diff_eq;
use machlearn::{MlError, SelectKBest, f_classif, f_regression};
use ndarray::array;

#[test]
fn selects_the_highest_scoring_classification_feature() {
    // Reference selection confirmed against
    // `sklearn.feature_selection.SelectKBest(score_func=f_classif, k=1)`
    // fitted on the same data.
    let records = array![
        [1.0, 10.0],
        [2.0, 9.0],
        [1.5, 11.0],
        [8.0, 1.0],
        [9.0, 2.0],
        [8.5, 0.5],
    ];
    let targets = array!["a", "a", "a", "b", "b", "b"];

    let fitted = SelectKBest::new(1)
        .fit(records.view(), targets.view(), f_classif)
        .unwrap();

    assert_eq!(fitted.selected_indices(), &[0]);
    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(transformed.ncols(), 1);
    assert_eq!(transformed.column(0).to_vec(), records.column(0).to_vec());
}

#[test]
fn selects_the_highest_scoring_regression_feature() {
    let records = array![[1.0, 5.0], [2.0, 3.0], [3.0, 5.0], [4.0, 3.0], [5.0, 5.0],];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];

    let fitted = SelectKBest::new(1)
        .fit(records.view(), targets.view(), f_regression)
        .unwrap();

    assert_eq!(fitted.selected_indices(), &[0]);
}

#[test]
fn k_larger_than_the_feature_count_keeps_every_feature() {
    let records = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let targets = array![1.0, 2.0, 3.0];

    let fitted = SelectKBest::new(10)
        .fit(records.view(), targets.view(), f_regression)
        .unwrap();

    assert_eq!(fitted.n_selected_features(), 2);
    assert_eq!(fitted.selected_indices(), &[0, 1]);
}

#[test]
fn k_zero_selects_no_features() {
    let records = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let targets = array![1.0, 2.0, 3.0];

    let fitted = SelectKBest::new(0)
        .fit(records.view(), targets.view(), f_regression)
        .unwrap();

    assert_eq!(fitted.n_selected_features(), 0);
    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(transformed.shape(), &[3, 0]);
}

#[test]
fn exposes_configuration() {
    let selector = SelectKBest::new(3);
    assert_eq!(selector.k(), 3);
}

#[test]
fn validates_matching_sample_counts() {
    let records = array![[1.0], [2.0], [3.0]];
    let targets = array![1.0, 2.0];

    assert_eq!(
        SelectKBest::new(1)
            .fit(records.view(), targets.view(), f_regression)
            .unwrap_err(),
        MlError::MismatchedSampleCount {
            feature_rows: 3,
            target_count: 2,
        }
    );
}

#[test]
fn validates_transform_features() {
    let records = array![[1.0, 2.0], [3.0, 4.0]];
    let targets = array![1.0, 2.0];
    let fitted = SelectKBest::new(1)
        .fit(records.view(), targets.view(), f_regression)
        .unwrap();

    assert_eq!(
        fitted.transform(array![[1.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
}

#[cfg(feature = "serde")]
#[test]
fn fitted_selector_round_trips_through_serde() {
    // Feature values are not perfectly correlated with the target, so
    // `f_regression`'s scores stay finite rather than producing the
    // `Infinity` a perfect fit would (which `serde_json` cannot round-trip:
    // it serializes to `null`, and `null` is not a valid `f64`).
    let records = array![[1.0, 5.0], [2.0, 3.0], [3.0, 6.0], [4.5, 3.5], [5.0, 5.0]];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let fitted = SelectKBest::new(1)
        .fit(records.view(), targets.view(), f_regression)
        .unwrap();

    let json = serde_json::to_string(&fitted).unwrap();
    let restored: machlearn::FittedSelectKBest = serde_json::from_str(&json).unwrap();

    // Compared field-by-field with a tolerance rather than `assert_eq!` on
    // the whole struct: `serde_json`'s text round trip for `f64` is not
    // guaranteed bit-exact in every case.
    assert_eq!(fitted.selected_indices(), restored.selected_indices());
    assert_eq!(fitted.n_features(), restored.n_features());
    for (left, right) in fitted.scores().iter().zip(restored.scores()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
