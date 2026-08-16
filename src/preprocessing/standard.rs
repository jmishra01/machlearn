// `ArrayView2` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::core::{Result, Transform, validate_feature_count, validate_features};

/// Configures feature-wise standardization.
///
/// The default estimator centers every feature and scales it to unit population
/// variance. Constant features use a scale of one and therefore remain finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandardScaler {
    with_mean: bool,
    with_std: bool,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }
}

impl StandardScaler {
    /// Enables or disables centering.
    #[must_use]
    pub const fn with_mean(mut self, enabled: bool) -> Self {
        self.with_mean = enabled;
        self
    }

    /// Enables or disables standard-deviation scaling.
    #[must_use]
    pub const fn with_std(mut self, enabled: bool) -> Self {
        self.with_std = enabled;
        self
    }

    /// Learns the per-feature offset and scale.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is empty or contains a non-finite value.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedStandardScaler> {
        validate_features(records)?;

        #[allow(clippy::cast_precision_loss)]
        let sample_count = records.nrows() as f64;
        let mut mean = Array1::zeros(records.ncols());
        let mut scale = Array1::ones(records.ncols());

        for (feature, column) in records.axis_iter(Axis(1)).enumerate() {
            let feature_mean = column.iter().sum::<f64>() / sample_count;
            if self.with_mean {
                mean[feature] = feature_mean;
            }
            if self.with_std {
                let variance = column
                    .iter()
                    .map(|value| {
                        let difference = value - feature_mean;
                        difference * difference
                    })
                    .sum::<f64>()
                    / sample_count;
                scale[feature] = nonzero_scale(variance.sqrt());
            }
        }

        Ok(FittedStandardScaler { mean, scale })
    }
}

/// Feature-wise parameters learned by [`StandardScaler`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedStandardScaler {
    mean: Array1<f64>,
    scale: Array1<f64>,
}

impl FittedStandardScaler {
    /// Returns the offset subtracted from each feature.
    #[must_use]
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// Returns the divisor applied to each feature.
    #[must_use]
    pub fn scale(&self) -> &Array1<f64> {
        &self.scale
    }

    /// Returns the number of features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.mean.len()
    }

    /// Standardizes a feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is empty, non-finite, or has a different
    /// feature count from the fitted data.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.validate_input(records)?;
        let mut output = records.to_owned();
        for mut row in output.rows_mut() {
            row -= &self.mean;
            row /= &self.scale;
        }
        Ok(output)
    }

    /// Restores standardized features to their original units.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is empty, non-finite, or has a different
    /// feature count from the fitted data.
    pub fn inverse_transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.validate_input(records)?;
        let mut output = records.to_owned();
        for mut row in output.rows_mut() {
            row *= &self.scale;
            row += &self.mean;
        }
        Ok(output)
    }

    fn validate_input(&self, records: ArrayView2<'_, f64>) -> Result<()> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedStandardScaler {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

#[allow(clippy::float_cmp)]
fn nonzero_scale(scale: f64) -> f64 {
    if scale == 0.0 { 1.0 } else { scale }
}
