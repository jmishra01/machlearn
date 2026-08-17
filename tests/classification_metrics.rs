//! Integration tests for classification accuracy and confusion matrices.

use approx::assert_abs_diff_eq;
use machlearn::{
    Averaging, ClassificationMetricOptions, MlError, ZeroDivision, accuracy_score,
    classification_report, confusion_matrix, f1_score, f1_score_with_options, precision_score,
    precision_score_with_options, recall_score, recall_score_with_options,
};
#[cfg(feature = "serde")]
use machlearn::{ClassificationReport, ConfusionMatrix};
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

#[test]
fn computes_macro_precision_recall_and_f1() {
    let actual = array![0, 1, 2, 0, 1, 2];
    let predicted = array![0, 2, 1, 0, 0, 2];

    assert_abs_diff_eq!(
        precision_score(actual.view(), predicted.view()).unwrap(),
        7.0 / 18.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        recall_score(actual.view(), predicted.view()).unwrap(),
        0.5,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        f1_score(actual.view(), predicted.view()).unwrap(),
        13.0 / 30.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn exposes_per_class_metrics_and_support() {
    let actual = array![0, 1, 2, 0, 1, 2];
    let predicted = array![0, 2, 1, 0, 0, 2];
    let report = classification_report(actual.view(), predicted.view()).unwrap();

    assert_eq!(report.n_classes(), 3);
    assert_abs_diff_eq!(report.accuracy(), 0.5);
    assert_eq!(*report.entries()[0].label(), 0);
    assert_abs_diff_eq!(report.entries()[0].precision(), 2.0 / 3.0);
    assert_abs_diff_eq!(report.entries()[0].recall(), 1.0);
    assert_abs_diff_eq!(report.entries()[0].f1(), 0.8);
    assert_eq!(report.entries()[0].support(), 2);
}

#[test]
fn supports_micro_and_weighted_averaging() {
    let actual = array![0, 0, 0, 1];
    let predicted = array![0, 0, 1, 1];
    let micro = ClassificationMetricOptions::new().with_averaging(Averaging::Micro);
    let weighted = ClassificationMetricOptions::new().with_averaging(Averaging::Weighted);

    for score in [
        precision_score_with_options(actual.view(), predicted.view(), micro).unwrap(),
        recall_score_with_options(actual.view(), predicted.view(), micro).unwrap(),
        f1_score_with_options(actual.view(), predicted.view(), micro).unwrap(),
    ] {
        assert_abs_diff_eq!(score, 0.75, epsilon = 1.0e-12);
    }
    assert_abs_diff_eq!(
        precision_score_with_options(actual.view(), predicted.view(), weighted).unwrap(),
        0.875,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        recall_score_with_options(actual.view(), predicted.view(), weighted).unwrap(),
        0.75,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        f1_score_with_options(actual.view(), predicted.view(), weighted).unwrap(),
        23.0 / 30.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn zero_division_behavior_is_explicit() {
    let actual = array!["a", "b"];
    let predicted = array!["a", "a"];
    let return_one = ClassificationMetricOptions::new().with_zero_division(ZeroDivision::One);
    let return_error = ClassificationMetricOptions::new().with_zero_division(ZeroDivision::Error);

    assert_abs_diff_eq!(
        precision_score(actual.view(), predicted.view()).unwrap(),
        0.25
    );
    assert_abs_diff_eq!(
        precision_score_with_options(actual.view(), predicted.view(), return_one).unwrap(),
        0.75
    );
    assert_eq!(
        precision_score_with_options(actual.view(), predicted.view(), return_error).unwrap_err(),
        MlError::UndefinedClassificationMetric {
            metric: "precision",
            class_index: 1,
        }
    );

    let predicted_only_class_actual = array!["a", "a"];
    let predicted_only_class_predictions = array!["a", "b"];
    assert_eq!(
        recall_score_with_options(
            predicted_only_class_actual.view(),
            predicted_only_class_predictions.view(),
            return_error,
        )
        .unwrap_err(),
        MlError::UndefinedClassificationMetric {
            metric: "recall",
            class_index: 1,
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

#[cfg(feature = "serde")]
#[test]
fn classification_report_round_trips_through_serde() {
    let report = classification_report(
        array!["cat".to_owned(), "dog".to_owned()].view(),
        array!["dog".to_owned(), "dog".to_owned()].view(),
    )
    .unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let restored: ClassificationReport<String> = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, report);
}
