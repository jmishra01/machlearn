// `ArrayView2` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::preprocessing::{FittedTransformer, TransformerEstimator};
use machlearn_core::core::{MlError, Result, Transform, validate_feature_count, validate_features};

const DEFAULT_THRESHOLD: f64 = 0.0;

/// Configures removal of low-variance features.
///
/// Every feature's population variance is computed across the training
/// rows; a feature is kept only when its variance is strictly greater than
/// `threshold`. The default threshold of `0.0` removes only exactly
/// constant features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarianceThreshold {
    threshold: f64,
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl VarianceThreshold {
    /// Creates a variance filter that removes exactly constant features
    /// (`threshold = 0.0`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Sets the minimum variance a feature must exceed to be kept.
    ///
    /// # Errors
    ///
    /// Returns an error when `threshold` is negative, NaN, or infinite.
    pub fn with_threshold(mut self, threshold: f64) -> Result<Self> {
        validate_threshold(threshold)?;
        self.threshold = threshold;
        Ok(self)
    }

    /// Returns the configured variance threshold.
    #[must_use]
    pub const fn threshold(self) -> f64 {
        self.threshold
    }

    /// Computes every feature's variance and selects those above
    /// `threshold`.
    ///
    /// # Errors
    ///
    /// Returns an error when `threshold` is negative, NaN, or infinite, or
    /// when features are empty or non-finite.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedVarianceThreshold> {
        validate_threshold(self.threshold)?;
        validate_features(records)?;

        let n_features = records.ncols();
        #[allow(clippy::cast_precision_loss)]
        let n_samples = records.nrows() as f64;
        let variances: Array1<f64> = Array1::from_iter(records.axis_iter(Axis(1)).map(|column| {
            let mean = column.sum() / n_samples;
            column
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / n_samples
        }));
        let selected_indices: Vec<usize> = (0..n_features)
            .filter(|&index| variances[index] > self.threshold)
            .collect();

        Ok(FittedVarianceThreshold {
            variances,
            selected_indices,
            n_features,
        })
    }
}

/// The per-feature variances and surviving columns learned by
/// [`VarianceThreshold`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedVarianceThreshold {
    variances: Array1<f64>,
    selected_indices: Vec<usize>,
    n_features: usize,
}

impl FittedVarianceThreshold {
    /// Returns every input feature's population variance, in original
    /// column order.
    #[must_use]
    pub const fn variances(&self) -> &Array1<f64> {
        &self.variances
    }

    /// Returns the original column indices that survived filtering, in
    /// ascending order.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of features that survived filtering.
    #[must_use]
    pub fn n_selected_features(&self) -> usize {
        self.selected_indices.len()
    }

    /// Keeps only the columns that survived filtering, preserving their
    /// original relative order.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;
        Ok(records.select(Axis(1), &self.selected_indices))
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedVarianceThreshold {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

impl TransformerEstimator for VarianceThreshold {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedVarianceThreshold {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}

fn validate_threshold(threshold: f64) -> Result<()> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(MlError::InvalidVarianceThreshold(threshold));
    }
    Ok(())
}
