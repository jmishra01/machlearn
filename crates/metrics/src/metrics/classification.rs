// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array2, ArrayView1, ArrayView2};

use machlearn_core::core::{MlError, Result};

/// Aggregation applied to multiclass precision, recall, and F1 scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Averaging {
    /// Average every class equally.
    #[default]
    Macro,
    /// Weight each class by its number of actual observations.
    Weighted,
    /// Aggregate all true positives, false positives, and false negatives.
    Micro,
}

/// Behavior when a per-class metric has a zero denominator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ZeroDivision {
    /// Return zero for the undefined per-class metric.
    #[default]
    Zero,
    /// Return one for the undefined per-class metric.
    One,
    /// Return [`MlError::UndefinedClassificationMetric`].
    Error,
}

/// Configuration for averaged classification metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassificationMetricOptions {
    averaging: Averaging,
    zero_division: ZeroDivision,
}

impl ClassificationMetricOptions {
    /// Creates macro-average options with undefined values replaced by zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            averaging: Averaging::Macro,
            zero_division: ZeroDivision::Zero,
        }
    }

    /// Sets the averaging method.
    #[must_use]
    pub const fn with_averaging(mut self, averaging: Averaging) -> Self {
        self.averaging = averaging;
        self
    }

    /// Sets zero-division behavior.
    #[must_use]
    pub const fn with_zero_division(mut self, behavior: ZeroDivision) -> Self {
        self.zero_division = behavior;
        self
    }

    /// Returns the configured averaging method.
    #[must_use]
    pub const fn averaging(self) -> Averaging {
        self.averaging
    }

    /// Returns the configured zero-division behavior.
    #[must_use]
    pub const fn zero_division(self) -> ZeroDivision {
        self.zero_division
    }
}

/// Precision, recall, F1, and support for one class.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassMetrics<Label> {
    label: Label,
    precision: f64,
    recall: f64,
    f1: f64,
    support: usize,
}

impl<Label> ClassMetrics<Label> {
    /// Returns the class label.
    #[must_use]
    pub const fn label(&self) -> &Label {
        &self.label
    }

    /// Returns precision for the class.
    #[must_use]
    pub const fn precision(&self) -> f64 {
        self.precision
    }

    /// Returns recall for the class.
    #[must_use]
    pub const fn recall(&self) -> f64 {
        self.recall
    }

    /// Returns F1 for the class.
    #[must_use]
    pub const fn f1(&self) -> f64 {
        self.f1
    }

    /// Returns the number of actual observations in the class.
    #[must_use]
    pub const fn support(&self) -> usize {
        self.support
    }
}

/// Per-class classification metrics in deterministic label order.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassificationReport<Label> {
    entries: Vec<ClassMetrics<Label>>,
    accuracy: f64,
}

impl<Label> ClassificationReport<Label> {
    /// Returns the per-class entries in sorted label order.
    #[must_use]
    pub fn entries(&self) -> &[ClassMetrics<Label>] {
        &self.entries
    }

    /// Returns the number of represented classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.entries.len()
    }

    /// Returns overall classification accuracy.
    #[must_use]
    pub const fn accuracy(&self) -> f64 {
        self.accuracy
    }
}

/// A class-by-class table of actual and predicted observation counts.
///
/// Rows correspond to actual classes and columns correspond to predicted
/// classes. [`ConfusionMatrix::classes`] defines the deterministic ordering used
/// by both axes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfusionMatrix<Label> {
    classes: Vec<Label>,
    counts: Array2<usize>,
}

impl<Label> ConfusionMatrix<Label> {
    /// Returns classes in row and column order.
    #[must_use]
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// Returns the count matrix, with actual classes on rows and predictions on
    /// columns.
    #[must_use]
    pub fn counts(&self) -> ArrayView2<'_, usize> {
        self.counts.view()
    }

    /// Returns the number of represented classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Returns the number of observations represented by the matrix.
    #[must_use]
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Returns the number of correctly classified observations.
    #[must_use]
    pub fn correct(&self) -> usize {
        self.counts.diag().iter().sum()
    }
}

/// Returns the fraction of predictions equal to their actual labels.
///
/// # Errors
///
/// Returns an error when the inputs are empty or have different lengths.
pub fn accuracy_score<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<f64>
where
    Label: PartialEq,
{
    validate_inputs(actual, predicted)?;
    let correct = actual
        .iter()
        .zip(predicted)
        .filter(|(actual, predicted)| actual == predicted)
        .count();
    #[allow(clippy::cast_precision_loss)]
    let result = correct as f64 / actual.len() as f64;
    Ok(result)
}

/// Builds a deterministic confusion matrix from actual and predicted labels.
///
/// The class list is the sorted union of labels appearing in either input.
/// Consequently, classes predicted but absent from the actual values are still
/// represented.
///
/// # Errors
///
/// Returns an error when the inputs are empty or have different lengths.
pub fn confusion_matrix<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<ConfusionMatrix<Label>>
where
    Label: Clone + Ord,
{
    validate_inputs(actual, predicted)?;

    let mut classes: Vec<_> = actual.iter().chain(predicted).cloned().collect();
    classes.sort_unstable();
    classes.dedup();

    let mut counts = Array2::zeros((classes.len(), classes.len()));
    for (actual, predicted) in actual.iter().zip(predicted) {
        // Both labels are present because `classes` is built from the union of
        // these same inputs. `partition_point` therefore returns their indices.
        let actual_index = classes.partition_point(|class| class < actual);
        let predicted_index = classes.partition_point(|class| class < predicted);
        counts[[actual_index, predicted_index]] += 1;
    }

    Ok(ConfusionMatrix { classes, counts })
}

/// Returns macro-averaged precision with zero used for undefined classes.
///
/// # Errors
///
/// Returns an error when the inputs are empty or have different lengths.
pub fn precision_score<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    precision_score_with_options(actual, predicted, ClassificationMetricOptions::default())
}

/// Returns precision using configurable averaging and zero-division behavior.
///
/// # Errors
///
/// Returns an error for invalid inputs or an undefined class when configured
/// with [`ZeroDivision::Error`].
pub fn precision_score_with_options<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
    options: ClassificationMetricOptions,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    averaged_score(actual, predicted, options, Metric::Precision)
}

/// Returns macro-averaged recall with zero used for undefined classes.
///
/// # Errors
///
/// Returns an error when the inputs are empty or have different lengths.
pub fn recall_score<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    recall_score_with_options(actual, predicted, ClassificationMetricOptions::default())
}

/// Returns recall using configurable averaging and zero-division behavior.
///
/// # Errors
///
/// Returns an error for invalid inputs or an undefined class when configured
/// with [`ZeroDivision::Error`].
pub fn recall_score_with_options<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
    options: ClassificationMetricOptions,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    averaged_score(actual, predicted, options, Metric::Recall)
}

/// Returns macro-averaged F1 with zero used for undefined classes.
///
/// # Errors
///
/// Returns an error when the inputs are empty or have different lengths.
pub fn f1_score<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    f1_score_with_options(actual, predicted, ClassificationMetricOptions::default())
}

/// Returns F1 using configurable averaging and zero-division behavior.
///
/// # Errors
///
/// Returns an error for invalid inputs or an undefined class when configured
/// with [`ZeroDivision::Error`].
pub fn f1_score_with_options<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
    options: ClassificationMetricOptions,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    averaged_score(actual, predicted, options, Metric::F1)
}

/// Builds a per-class report using zero for undefined metrics.
///
/// # Errors
///
/// Returns an error when the inputs are empty or have different lengths.
pub fn classification_report<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<ClassificationReport<Label>>
where
    Label: Clone + Ord,
{
    classification_report_with_zero_division(actual, predicted, ZeroDivision::Zero)
}

/// Builds a per-class report with explicit zero-division behavior.
///
/// # Errors
///
/// Returns an error for invalid inputs or an undefined class when configured
/// with [`ZeroDivision::Error`].
pub fn classification_report_with_zero_division<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
    zero_division: ZeroDivision,
) -> Result<ClassificationReport<Label>>
where
    Label: Clone + Ord,
{
    let matrix = confusion_matrix(actual, predicted)?;
    report_from_matrix(matrix, zero_division)
}

#[derive(Clone, Copy)]
enum Metric {
    Precision,
    Recall,
    F1,
}

fn averaged_score<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
    options: ClassificationMetricOptions,
    metric: Metric,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    let matrix = confusion_matrix(actual, predicted)?;
    if options.averaging == Averaging::Micro {
        #[allow(clippy::cast_precision_loss)]
        return Ok(matrix.correct() as f64 / matrix.total() as f64);
    }

    let report = report_from_matrix(matrix, options.zero_division)?;
    let score = |entry: &ClassMetrics<Label>| match metric {
        Metric::Precision => entry.precision,
        Metric::Recall => entry.recall,
        Metric::F1 => entry.f1,
    };

    #[allow(clippy::cast_precision_loss)]
    let result = match options.averaging {
        Averaging::Macro => {
            report.entries.iter().map(score).sum::<f64>() / report.entries.len() as f64
        }
        Averaging::Weighted => {
            report
                .entries
                .iter()
                .map(|entry| score(entry) * entry.support as f64)
                .sum::<f64>()
                / report
                    .entries
                    .iter()
                    .map(|entry| entry.support)
                    .sum::<usize>() as f64
        }
        Averaging::Micro => unreachable!("micro averaging returned before report construction"),
    };
    Ok(result)
}

fn report_from_matrix<Label>(
    matrix: ConfusionMatrix<Label>,
    zero_division: ZeroDivision,
) -> Result<ClassificationReport<Label>> {
    let total = matrix.total();
    let correct = matrix.correct();
    let (classes, counts) = (matrix.classes, matrix.counts);
    let mut entries = Vec::with_capacity(classes.len());

    for (class_index, label) in classes.into_iter().enumerate() {
        let true_positive = counts[[class_index, class_index]];
        let predicted_count: usize = counts.column(class_index).iter().sum();
        let support: usize = counts.row(class_index).iter().sum();
        let precision = ratio(
            true_positive,
            predicted_count,
            "precision",
            class_index,
            zero_division,
        )?;
        let recall = ratio(true_positive, support, "recall", class_index, zero_division)?;
        let f1 = harmonic_mean(precision, recall);
        entries.push(ClassMetrics {
            label,
            precision,
            recall,
            f1,
            support,
        });
    }

    #[allow(clippy::cast_precision_loss)]
    let accuracy = correct as f64 / total as f64;
    Ok(ClassificationReport { entries, accuracy })
}

fn ratio(
    numerator: usize,
    denominator: usize,
    metric: &'static str,
    class_index: usize,
    zero_division: ZeroDivision,
) -> Result<f64> {
    if denominator == 0 {
        return match zero_division {
            ZeroDivision::Zero => Ok(0.0),
            ZeroDivision::One => Ok(1.0),
            ZeroDivision::Error => Err(MlError::UndefinedClassificationMetric {
                metric,
                class_index,
            }),
        };
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(numerator as f64 / denominator as f64)
}

#[allow(clippy::float_cmp)]
fn harmonic_mean(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn validate_inputs<Label>(
    actual: ArrayView1<'_, Label>,
    predicted: ArrayView1<'_, Label>,
) -> Result<()> {
    if actual.is_empty() {
        return Err(MlError::EmptyMetricInput);
    }
    if actual.len() != predicted.len() {
        return Err(MlError::MismatchedMetricInput {
            actual: actual.len(),
            predicted: predicted.len(),
        });
    }
    Ok(())
}
