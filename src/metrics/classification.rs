// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::core::{MlError, Result};

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
