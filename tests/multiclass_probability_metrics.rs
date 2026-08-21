//! Integration tests for multiclass probability metrics.

use approx::assert_abs_diff_eq;
use machlearn::{MlError, multiclass_log_loss, roc_auc_score_ovr};
use ndarray::array;

fn dataset() -> (
    ndarray::Array1<&'static str>,
    ndarray::Array2<f64>,
    [&'static str; 3],
) {
    let actual = array!["a", "b", "c", "a", "b", "c", "a", "c", "b", "a"];
    let probabilities = array![
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.3, 0.3, 0.4],
        [0.5, 0.2, 0.3],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
        [0.4, 0.4, 0.2],
        [0.3, 0.2, 0.5],
        [0.25, 0.55, 0.2],
        [0.55, 0.25, 0.2],
    ];
    (actual, probabilities, ["a", "b", "c"])
}

#[test]
fn matches_reference_multiclass_log_loss() {
    // Reference value confirmed against
    // `sklearn.metrics.log_loss(y_true, y_proba, labels=["a","b","c"])` on
    // the same data.
    let (actual, probabilities, classes) = dataset();

    let loss = multiclass_log_loss(actual.view(), probabilities.view(), &classes).unwrap();

    assert_abs_diff_eq!(loss, 0.722_796_054_313_947_8, epsilon = 1.0e-12);
}

#[test]
fn matches_reference_one_vs_rest_roc_auc() {
    // Reference value confirmed against
    // `sklearn.metrics.roc_auc_score(y_true, y_proba, multi_class="ovr",
    // average="macro", labels=["a","b","c"])` on the same data.
    let (actual, probabilities, classes) = dataset();

    let auc = roc_auc_score_ovr(actual.view(), probabilities.view(), &classes).unwrap();

    assert_abs_diff_eq!(auc, 0.992_063_492_063_492_1, epsilon = 1.0e-12);
}

#[test]
fn perfect_probabilities_score_zero_loss_and_perfect_auc() {
    let actual = array!["a", "b", "c"];
    let probabilities = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let classes = ["a", "b", "c"];

    let loss = multiclass_log_loss(actual.view(), probabilities.view(), &classes).unwrap();
    assert!(loss.is_finite());
    assert!(loss <= f64::EPSILON * 4.0);

    let auc = roc_auc_score_ovr(actual.view(), probabilities.view(), &classes).unwrap();
    assert_abs_diff_eq!(auc, 1.0, epsilon = 1.0e-12);
}

#[test]
fn log_loss_rejects_an_unknown_label() {
    let actual = array!["a", "z"];
    let probabilities = array![[0.6, 0.4], [0.5, 0.5]];
    let classes = ["a", "b"];

    assert_eq!(
        multiclass_log_loss(actual.view(), probabilities.view(), &classes).unwrap_err(),
        MlError::UnknownLabel { index: 1 }
    );
}

#[test]
fn rejects_a_mismatched_class_count() {
    let (actual, probabilities, _classes) = dataset();
    let wrong_classes = ["a", "b"];

    assert_eq!(
        multiclass_log_loss(actual.view(), probabilities.view(), &wrong_classes).unwrap_err(),
        MlError::MismatchedClassCount {
            expected: 2,
            actual: 3,
        }
    );
    assert_eq!(
        roc_auc_score_ovr(actual.view(), probabilities.view(), &wrong_classes).unwrap_err(),
        MlError::MismatchedClassCount {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn roc_auc_ovr_requires_at_least_two_classes() {
    let actual = array!["a", "a", "a"];
    let probabilities = array![[1.0], [1.0], [1.0]];
    let classes = ["a"];

    assert_eq!(
        roc_auc_score_ovr(actual.view(), probabilities.view(), &classes).unwrap_err(),
        MlError::InsufficientClasses {
            required: 2,
            actual: 1,
        }
    );
}

#[test]
fn rejects_non_finite_and_out_of_range_probabilities() {
    let actual = array!["a", "b"];
    let classes = ["a", "b"];

    assert_eq!(
        multiclass_log_loss(
            actual.view(),
            array![[f64::NAN, 0.0], [0.5, 0.5]].view(),
            &classes
        )
        .unwrap_err(),
        MlError::NonFiniteProbability { index: 0 }
    );
    assert_eq!(
        roc_auc_score_ovr(
            actual.view(),
            array![[1.5, -0.5], [0.5, 0.5]].view(),
            &classes
        )
        .unwrap_err(),
        MlError::InvalidProbability {
            index: 0,
            value: 1.5
        }
    );
}

#[test]
fn rejects_empty_and_different_length_inputs() {
    let empty_labels = ndarray::Array1::<&str>::from_vec(Vec::new());
    let empty_probabilities = ndarray::Array2::<f64>::zeros((0, 2));
    let classes = ["a", "b"];

    assert_eq!(
        multiclass_log_loss(empty_labels.view(), empty_probabilities.view(), &classes).unwrap_err(),
        MlError::EmptyMetricInput
    );
    assert_eq!(
        roc_auc_score_ovr(array!["a", "b"].view(), array![[0.5, 0.5]].view(), &classes)
            .unwrap_err(),
        MlError::MismatchedMetricInput {
            actual: 2,
            predicted: 1,
        }
    );
}
