use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::core::{MlError, Result, validate_features};

/// A validated dense feature matrix paired with one target per sample.
///
/// Features are represented as `f64` for the initial API. Targets are generic,
/// allowing regression values, integer class labels, or application-specific
/// label types.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dataset<Target = f64> {
    records: Array2<f64>,
    targets: Array1<Target>,
}

impl<Target> Dataset<Target> {
    /// Creates a dataset after validating its dimensions and feature values.
    ///
    /// # Errors
    ///
    /// Returns an error when the dataset has no samples or features, when its
    /// target count differs from its row count, or when a feature is not finite.
    pub fn new(records: Array2<f64>, targets: Array1<Target>) -> Result<Self> {
        validate_features(records.view())?;
        validate_target_count(records.nrows(), targets.len())?;
        Ok(Self { records, targets })
    }

    /// Returns a read-only view of the feature matrix.
    #[must_use]
    pub fn records(&self) -> ArrayView2<'_, f64> {
        self.records.view()
    }

    /// Returns a read-only view of the target vector.
    #[must_use]
    pub fn targets(&self) -> ArrayView1<'_, Target> {
        self.targets.view()
    }

    /// Returns the number of samples in the dataset.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.records.nrows()
    }

    /// Returns the number of features in the dataset.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.records.ncols()
    }

    /// Returns `(samples, features)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.records.dim()
    }

    /// Consumes the dataset and returns its feature and target arrays.
    #[must_use]
    pub fn into_parts(self) -> (Array2<f64>, Array1<Target>) {
        (self.records, self.targets)
    }
}

fn validate_target_count(feature_rows: usize, target_count: usize) -> Result<()> {
    if feature_rows != target_count {
        return Err(MlError::MismatchedSampleCount {
            feature_rows,
            target_count,
        });
    }
    Ok(())
}
