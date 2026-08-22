// `ArrayView1` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView1};

use machlearn_core::core::{MlError, Result, Transform};

/// Configures deterministic categorical label encoding.
///
/// Classes are ordered using [`Ord`], so a fitted encoder assigns the same
/// integer to a class regardless of the order in which training labels appear.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LabelEncoder;

impl LabelEncoder {
    /// Creates a label encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Learns the sorted set of unique labels.
    ///
    /// # Errors
    ///
    /// Returns an error when `labels` is empty.
    pub fn fit<Label>(&self, labels: ArrayView1<'_, Label>) -> Result<FittedLabelEncoder<Label>>
    where
        Label: Clone + Ord,
    {
        if labels.is_empty() {
            return Err(MlError::EmptyTargets);
        }

        let mut classes = labels.to_vec();
        classes.sort_unstable();
        classes.dedup();
        Ok(FittedLabelEncoder { classes })
    }
}

/// The class table learned by [`LabelEncoder`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedLabelEncoder<Label> {
    classes: Vec<Label>,
}

impl<Label> FittedLabelEncoder<Label> {
    /// Returns classes in their encoded-index order.
    #[must_use]
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// Returns the number of learned classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

impl<Label> FittedLabelEncoder<Label>
where
    Label: Ord,
{
    /// Encodes labels as zero-based class indices.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::UnknownLabel`] at the first label that was not seen
    /// during fitting.
    pub fn transform(&self, labels: ArrayView1<'_, Label>) -> Result<Array1<usize>> {
        labels
            .iter()
            .enumerate()
            .map(|(index, label)| {
                self.classes
                    .binary_search(label)
                    .map_err(|_| MlError::UnknownLabel { index })
            })
            .collect()
    }
}

impl<Label> FittedLabelEncoder<Label>
where
    Label: Clone,
{
    /// Converts class indices back to their original labels.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::InvalidClassIndex`] at the first index outside the
    /// fitted class table.
    pub fn inverse_transform(&self, encoded: ArrayView1<'_, usize>) -> Result<Array1<Label>> {
        encoded
            .iter()
            .enumerate()
            .map(|(position, &class_index)| {
                self.classes
                    .get(class_index)
                    .cloned()
                    .ok_or(MlError::InvalidClassIndex {
                        position,
                        class_index,
                        class_count: self.classes.len(),
                    })
            })
            .collect()
    }
}

impl<'a, Label> Transform<ArrayView1<'a, Label>> for FittedLabelEncoder<Label>
where
    Label: Ord,
{
    type Output = Array1<usize>;

    fn transform(&self, input: ArrayView1<'a, Label>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}
