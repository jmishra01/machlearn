//! Integration tests for binary probability metrics.

use approx::assert_abs_diff_eq;
use machlearn::{MlError, binary_log_loss, roc_auc_score};
use ndarray::{Array1, array};

#[test]
fn matches_reference_log_loss_and_roc_auc() {
    let actual = array![0, 0, 1, 1];
    let probabilities = array![0.1, 0.4, 0.35, 0.8];

    assert_abs_diff_eq!(
        binary_log_loss(actual.view(), probabilities.view(), &1).unwrap(),
        0.472_287_953_809_176_15,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        roc_auc_score(actual.view(), probabilities.view(), &1).unwrap(),
        0.75,
        epsilon = 1.0e-12
    );
}

#[test]
fn roc_auc_handles_ties_with_half_credit() {
    assert_abs_diff_eq!(
        roc_auc_score(array!["no", "yes"].view(), array![0.5, 0.5].view(), &"yes").unwrap(),
        0.5,
        epsilon = 1.0e-12
    );
}

#[test]
fn roc_auc_identifies_perfect_and_reversed_rankings() {
    let actual = array![false, false, true, true];
    assert_abs_diff_eq!(
        roc_auc_score(actual.view(), array![0.1, 0.2, 0.8, 0.9].view(), &true).unwrap(),
        1.0
    );
    assert_abs_diff_eq!(
        roc_auc_score(actual.view(), array![0.9, 0.8, 0.2, 0.1].view(), &true).unwrap(),
        0.0
    );
}

#[test]
fn log_loss_clips_exact_probability_endpoints() {
    let perfect = binary_log_loss(array![0, 1].view(), array![0.0, 1.0].view(), &1).unwrap();
    let confidently_wrong =
        binary_log_loss(array![0, 1].view(), array![1.0, 0.0].view(), &1).unwrap();

    assert!(perfect.is_finite());
    assert!(perfect <= f64::EPSILON * 2.0);
    assert!(confidently_wrong.is_finite());
    assert!(confidently_wrong > 30.0);
}

#[test]
fn rejects_non_finite_and_out_of_range_probabilities() {
    assert_eq!(
        binary_log_loss(array![0, 1].view(), array![0.2, f64::NAN].view(), &1).unwrap_err(),
        MlError::NonFiniteProbability { index: 1 }
    );
    for (value, expected_index) in [(-0.1, 0), (1.1, 0)] {
        assert_eq!(
            roc_auc_score(array![0, 1].view(), array![value, 0.5].view(), &1).unwrap_err(),
            MlError::InvalidProbability {
                index: expected_index,
                value,
            }
        );
    }
}

#[test]
fn requires_exactly_two_observed_classes() {
    assert_eq!(
        binary_log_loss(array![1, 1].view(), array![0.8, 0.9].view(), &1).unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 1 }
    );
    assert_eq!(
        roc_auc_score(array![0, 1, 2].view(), array![0.1, 0.5, 0.9].view(), &2).unwrap_err(),
        MlError::ExpectedBinaryTargets { class_count: 3 }
    );
}

#[test]
fn requires_the_positive_label_to_be_observed() {
    assert_eq!(
        roc_auc_score(array![0, 1].view(), array![0.1, 0.9].view(), &2).unwrap_err(),
        MlError::PositiveLabelNotFound
    );
}

#[test]
fn rejects_empty_and_different_length_inputs() {
    let empty_labels = Array1::<u8>::zeros(0);
    let empty_probabilities = Array1::<f64>::zeros(0);
    assert_eq!(
        binary_log_loss(empty_labels.view(), empty_probabilities.view(), &1).unwrap_err(),
        MlError::EmptyMetricInput
    );
    assert_eq!(
        roc_auc_score(array![0, 1].view(), array![0.1].view(), &1).unwrap_err(),
        MlError::MismatchedMetricInput {
            actual: 2,
            predicted: 1,
        }
    );
}
