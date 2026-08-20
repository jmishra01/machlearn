// `ArrayView1` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array2, ArrayView1};

use crate::core::{MlError, Result, Transform};

/// Configures deterministic one-hot (or dummy) encoding of a categorical
/// column.
///
/// Classes are ordered using [`Ord`], so a fitted encoder assigns the same
/// output column to a class regardless of the order in which training
/// labels appear, matching [`crate::LabelEncoder`]'s convention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OneHotEncoder {
    drop_first: bool,
}

impl OneHotEncoder {
    /// Creates a one-hot encoder that emits one indicator column per class.
    #[must_use]
    pub const fn new() -> Self {
        Self { drop_first: false }
    }

    /// Sets whether the first sorted class's indicator column is dropped.
    ///
    /// Dropping it produces the "dummy encoding" convention: `n_classes -
    /// 1` columns instead of `n_classes`, avoiding perfect collinearity
    /// with an intercept term. The dropped class is represented implicitly
    /// by an all-zero row.
    #[must_use]
    pub const fn with_drop_first(mut self, drop_first: bool) -> Self {
        self.drop_first = drop_first;
        self
    }

    /// Returns whether the first sorted class's column is dropped.
    #[must_use]
    pub const fn drop_first(self) -> bool {
        self.drop_first
    }

    /// Learns the sorted set of unique classes.
    ///
    /// # Errors
    ///
    /// Returns an error when `labels` is empty.
    pub fn fit<Label>(&self, labels: ArrayView1<'_, Label>) -> Result<FittedOneHotEncoder<Label>>
    where
        Label: Clone + Ord,
    {
        if labels.is_empty() {
            return Err(MlError::EmptyTargets);
        }

        let mut classes = labels.to_vec();
        classes.sort_unstable();
        classes.dedup();
        Ok(FittedOneHotEncoder {
            classes,
            drop_first: self.drop_first,
        })
    }
}

/// The class table learned by [`OneHotEncoder`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedOneHotEncoder<Label> {
    classes: Vec<Label>,
    drop_first: bool,
}

impl<Label> FittedOneHotEncoder<Label> {
    /// Returns classes in their encoded-column order.
    ///
    /// When [`Self::drop_first`] is set, the first class here has no
    /// indicator column of its own; it is the row of all zeros.
    #[must_use]
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// Returns the number of learned classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Returns the number of columns [`Self::transform`] produces.
    #[must_use]
    pub fn n_output_columns(&self) -> usize {
        if self.drop_first {
            self.n_classes().saturating_sub(1)
        } else {
            self.n_classes()
        }
    }

    /// Returns whether the first sorted class's column is dropped.
    #[must_use]
    pub const fn drop_first(&self) -> bool {
        self.drop_first
    }
}

impl<Label> FittedOneHotEncoder<Label>
where
    Label: Ord,
{
    /// Encodes labels as indicator columns, one row per label.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::UnknownLabel`] at the first label that was not
    /// seen during fitting.
    pub fn transform(&self, labels: ArrayView1<'_, Label>) -> Result<Array2<f64>> {
        let mut output = Array2::zeros((labels.len(), self.n_output_columns()));
        for (row, label) in labels.iter().enumerate() {
            let class_index = self
                .classes
                .binary_search(label)
                .map_err(|_| MlError::UnknownLabel { index: row })?;
            if self.drop_first {
                if class_index == 0 {
                    continue;
                }
                output[[row, class_index - 1]] = 1.0;
            } else {
                output[[row, class_index]] = 1.0;
            }
        }
        Ok(output)
    }
}

impl<'a, Label> Transform<ArrayView1<'a, Label>> for FittedOneHotEncoder<Label>
where
    Label: Ord,
{
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView1<'a, Label>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}
