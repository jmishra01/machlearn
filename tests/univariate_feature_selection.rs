//! Integration tests for univariate F-test feature scoring.

use approx::assert_abs_diff_eq;
use machlearn::{MlError, f_classif, f_regression};
use ndarray::array;

#[test]
fn f_classif_matches_a_reference_solution() {
    // Reference values confirmed against
    // `sklearn.feature_selection.f_classif` on the same data.
    let records = array![
        [1.0, 10.0],
        [2.0, 9.0],
        [1.5, 11.0],
        [8.0, 1.0],
        [9.0, 2.0],
        [8.5, 0.5],
    ];
    let targets = array!["a", "a", "a", "b", "b", "b"];

    let scores = f_classif(records.view(), targets.view()).unwrap();

    assert_abs_diff_eq!(scores[0], 294.0, epsilon = 1.0e-6);
    assert_abs_diff_eq!(scores[1], 147.842_105_263_157_9, epsilon = 1.0e-6);
}

#[test]
fn f_classif_rejects_a_single_class() {
    let records = array![[1.0], [2.0], [3.0]];
    let targets = array!["a", "a", "a"];

    assert_eq!(
        f_classif(records.view(), targets.view()).unwrap_err(),
        MlError::InsufficientClasses {
            required: 2,
            actual: 1,
        }
    );
}

#[test]
fn f_classif_validates_matching_sample_counts() {
    let records = array![[1.0], [2.0], [3.0]];
    let targets = array!["a", "b"];

    assert_eq!(
        f_classif(records.view(), targets.view()).unwrap_err(),
        MlError::MismatchedSampleCount {
            feature_rows: 3,
            target_count: 2,
        }
    );
}

#[test]
fn f_regression_matches_a_reference_solution() {
    // Reference value confirmed against
    // `sklearn.feature_selection.f_regression` on the same data.
    let records = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let targets = array![2.0, 4.0, 5.0, 4.5, 5.5];

    let scores = f_regression(records.view(), targets.view()).unwrap();

    assert_abs_diff_eq!(scores[0], 10.074_626_865_671_641, epsilon = 1.0e-6);
}

#[test]
fn f_regression_scores_an_uncorrelated_feature_near_zero() {
    let records = array![[1.0, 5.0], [2.0, 3.0], [3.0, 5.0], [4.0, 3.0], [5.0, 5.0]];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];

    let scores = f_regression(records.view(), targets.view()).unwrap();

    assert!(scores[0] > scores[1]);
}

#[test]
fn f_regression_validates_matching_sample_counts() {
    let records = array![[1.0], [2.0], [3.0]];
    let targets = array![1.0, 2.0];

    assert_eq!(
        f_regression(records.view(), targets.view()).unwrap_err(),
        MlError::MismatchedSampleCount {
            feature_rows: 3,
            target_count: 2,
        }
    );
}
