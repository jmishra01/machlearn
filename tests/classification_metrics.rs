//! Integration tests for classification accuracy and confusion matrices.

use approx::assert_abs_diff_eq;
#[cfg(feature = "serde")]
use machlearn::ConfusionMatrix;
use machlearn::{MlError, accuracy_score, confusion_matrix};
use ndarray::{Array1, array};

#[test]
fn computes_accuracy_for_string_labels() {
    let actual = array!["cat", "dog", "cat", "bird"];
    let predicted = array!["cat", "cat", "dog", "bird"];

    assert_abs_diff_eq!(
        accuracy_score(actual.view(), predicted.view()).unwrap(),
        0.5,
        epsilon = 1.0e-12
    );
}

#[test]
fn builds_a_sorted_actual_by_predicted_matrix() {
    let actual = array!["cat", "dog", "cat", "bird"];
    let predicted = array!["cat", "cat", "dog", "bird"];
    let matrix = confusion_matrix(actual.view(), predicted.view()).unwrap();

    assert_eq!(matrix.classes(), &["bird", "cat", "dog"]);
    assert_eq!(
        matrix.counts(),
        array![[1_usize, 0, 0], [0, 1, 1], [0, 1, 0]].view()
    );
    assert_eq!(matrix.n_classes(), 3);
    assert_eq!(matrix.total(), 4);
    assert_eq!(matrix.correct(), 2);
}

#[test]
fn includes_classes_that_appear_only_in_predictions() {
    let matrix = confusion_matrix(array![0, 0].view(), array![0, 1].view()).unwrap();

    assert_eq!(matrix.classes(), &[0, 1]);
    assert_eq!(matrix.counts(), array![[1_usize, 1], [0, 0]].view());
}

#[test]
fn class_order_is_independent_of_observation_order() {
    let first = confusion_matrix(array![3, 1, 2].view(), array![2, 1, 3].view()).unwrap();
    let second = confusion_matrix(array![2, 3, 1].view(), array![3, 2, 1].view()).unwrap();

    assert_eq!(first.classes(), &[1, 2, 3]);
    assert_eq!(second.classes(), first.classes());
    assert_eq!(second.counts(), first.counts());
}

#[test]
fn rejects_empty_inputs() {
    let empty = Array1::<u8>::zeros(0);
    assert_eq!(
        accuracy_score(empty.view(), empty.view()).unwrap_err(),
        MlError::EmptyMetricInput
    );
    assert_eq!(
        confusion_matrix(empty.view(), empty.view()).unwrap_err(),
        MlError::EmptyMetricInput
    );
}

#[test]
fn rejects_different_input_lengths() {
    assert_eq!(
        accuracy_score(array![1, 2].view(), array![1].view()).unwrap_err(),
        MlError::MismatchedMetricInput {
            actual: 2,
            predicted: 1,
        }
    );
}

#[cfg(feature = "serde")]
#[test]
fn confusion_matrix_round_trips_through_serde() {
    let matrix = confusion_matrix(
        array!["cat".to_owned(), "dog".to_owned()].view(),
        array!["dog".to_owned(), "dog".to_owned()].view(),
    )
    .unwrap();
    let json = serde_json::to_string(&matrix).unwrap();
    let restored: ConfusionMatrix<String> = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, matrix);
}
