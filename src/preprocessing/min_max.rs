// `ArrayView2` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::core::{MlError, Result, Transform, validate_feature_count, validate_features};

/// Configures feature-wise linear scaling to a target range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxScaler {
    minimum: f64,
    maximum: f64,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self {
            minimum: 0.0,
            maximum: 1.0,
        }
    }
}

impl MinMaxScaler {
    /// Creates a scaler targeting `(minimum, maximum)`.
    ///
    /// # Errors
    ///
    /// Returns an error unless both bounds are finite and `minimum < maximum`.
    pub fn new(minimum: f64, maximum: f64) -> Result<Self> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum >= maximum {
            return Err(MlError::InvalidFeatureRange { minimum, maximum });
        }
        Ok(Self { minimum, maximum })
    }

    /// Returns the configured output range.
    #[must_use]
    pub const fn feature_range(self) -> (f64, f64) {
        (self.minimum, self.maximum)
    }

    /// Learns a linear scaling for every feature.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is empty or contains a non-finite value.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedMinMaxScaler> {
        validate_features(records)?;

        let mut data_min = Array1::zeros(records.ncols());
        let mut data_max = Array1::zeros(records.ncols());
        let mut scale = Array1::zeros(records.ncols());
        let mut offset = Array1::zeros(records.ncols());
        let output_span = self.maximum - self.minimum;

        for (feature, column) in records.axis_iter(Axis(1)).enumerate() {
            let minimum = column.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let data_span = nonzero_span(maximum - minimum);

            data_min[feature] = minimum;
            data_max[feature] = maximum;
            scale[feature] = output_span / data_span;
            offset[feature] = self.minimum - minimum * scale[feature];
        }

        Ok(FittedMinMaxScaler {
            data_min,
            data_max,
            scale,
            offset,
            feature_range: (self.minimum, self.maximum),
        })
    }
}

/// Feature-wise parameters learned by [`MinMaxScaler`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedMinMaxScaler {
    data_min: Array1<f64>,
    data_max: Array1<f64>,
    scale: Array1<f64>,
    offset: Array1<f64>,
    feature_range: (f64, f64),
}

impl FittedMinMaxScaler {
    /// Returns the minimum observed in each input feature.
    #[must_use]
    pub fn data_min(&self) -> &Array1<f64> {
        &self.data_min
    }

    /// Returns the maximum observed in each input feature.
    #[must_use]
    pub fn data_max(&self) -> &Array1<f64> {
        &self.data_max
    }

    /// Returns the configured output range.
    #[must_use]
    pub const fn feature_range(&self) -> (f64, f64) {
        self.feature_range
    }

    /// Returns the number of features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.scale.len()
    }

    /// Scales a feature matrix to the configured range.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is empty, non-finite, or has a different
    /// feature count from the fitted data.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.validate_input(records)?;
        let mut output = records.to_owned();
        for mut row in output.rows_mut() {
            row *= &self.scale;
            row += &self.offset;
        }
        Ok(output)
    }

    /// Restores scaled features to their original units.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is empty, non-finite, or has a different
    /// feature count from the fitted data.
    pub fn inverse_transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.validate_input(records)?;
        let mut output = records.to_owned();
        for mut row in output.rows_mut() {
            row -= &self.offset;
            row /= &self.scale;
        }
        Ok(output)
    }

    fn validate_input(&self, records: ArrayView2<'_, f64>) -> Result<()> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedMinMaxScaler {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

#[allow(clippy::float_cmp)]
fn nonzero_span(span: f64) -> f64 {
    if span == 0.0 { 1.0 } else { span }
}
