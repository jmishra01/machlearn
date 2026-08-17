// `ArrayView2` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::core::{
    MlError, Result, Transform, validate_feature_count, validate_features_allow_nan,
};

/// Strategy used to replace `NaN` feature values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImputationStrategy {
    /// Replace missing values with the mean of observed values in their column.
    Mean,
    /// Replace missing values with the median of observed values in their column.
    Median,
    /// Replace every missing value with a fixed finite value.
    Constant(f64),
}

/// Learns feature-wise values for replacing missing data.
///
/// `NaN` represents a missing value. Positive and negative infinity are always
/// rejected. The default strategy is [`ImputationStrategy::Mean`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimpleImputer {
    strategy: ImputationStrategy,
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self::mean()
    }
}

impl SimpleImputer {
    /// Creates a mean imputer.
    #[must_use]
    pub const fn mean() -> Self {
        Self {
            strategy: ImputationStrategy::Mean,
        }
    }

    /// Creates a median imputer.
    #[must_use]
    pub const fn median() -> Self {
        Self {
            strategy: ImputationStrategy::Median,
        }
    }

    /// Creates an imputer that uses a fixed value.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is NaN or infinite.
    pub fn constant(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(MlError::InvalidImputationConstant(value));
        }
        Ok(Self {
            strategy: ImputationStrategy::Constant(value),
        })
    }

    /// Returns the configured strategy.
    #[must_use]
    pub const fn strategy(self) -> ImputationStrategy {
        self.strategy
    }

    /// Learns one replacement value per feature.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input, infinity, a non-finite learned
    /// statistic, or an entirely missing column when using mean or median.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedSimpleImputer> {
        validate_features_allow_nan(records)?;

        let mut fill_values = Array1::zeros(records.ncols());
        for (column_index, column) in records.axis_iter(Axis(1)).enumerate() {
            fill_values[column_index] = match self.strategy {
                ImputationStrategy::Constant(value) => value,
                ImputationStrategy::Mean => mean(column_index, column.iter().copied())?,
                ImputationStrategy::Median => median(column_index, column.iter().copied())?,
            };
        }

        Ok(FittedSimpleImputer { fill_values })
    }
}

/// Feature-wise replacement values learned by [`SimpleImputer`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedSimpleImputer {
    fill_values: Array1<f64>,
}

impl FittedSimpleImputer {
    /// Returns the learned replacement value for every feature.
    #[must_use]
    pub fn fill_values(&self) -> &Array1<f64> {
        &self.fill_values
    }

    /// Returns the number of features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.fill_values.len()
    }

    /// Replaces every `NaN` with its feature's learned value.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input, infinity, or a different feature count
    /// from the fitted data.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features_allow_nan(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;

        let mut output = records.to_owned();
        for ((_, column), value) in output.indexed_iter_mut() {
            if value.is_nan() {
                *value = self.fill_values[column];
            }
        }
        Ok(output)
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedSimpleImputer {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

fn observed(column: usize, values: impl Iterator<Item = f64>) -> Result<Vec<f64>> {
    let values: Vec<_> = values.filter(|value| !value.is_nan()).collect();
    if values.is_empty() {
        return Err(MlError::AllValuesMissing { column });
    }
    Ok(values)
}

fn mean(column: usize, values: impl Iterator<Item = f64>) -> Result<f64> {
    let values = observed(column, values)?;
    #[allow(clippy::cast_precision_loss)]
    let count = values.len() as f64;
    let statistic = values.iter().map(|value| value / count).sum::<f64>();
    validate_statistic(column, statistic)
}

fn median(column: usize, values: impl Iterator<Item = f64>) -> Result<f64> {
    let mut values = observed(column, values)?;
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    let statistic = if values.len() % 2 == 0 {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    };
    validate_statistic(column, statistic)
}

fn validate_statistic(column: usize, statistic: f64) -> Result<f64> {
    if !statistic.is_finite() {
        return Err(MlError::NonFiniteImputationStatistic { column });
    }
    Ok(statistic)
}
