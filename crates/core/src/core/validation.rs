// `ArrayView2` is a lightweight view descriptor; accepting it by value keeps
// call sites consistent with the rest of the ndarray ecosystem.
#![allow(clippy::needless_pass_by_value)]

use ndarray::ArrayView2;

use crate::core::{MlError, Result};

/// Validates that `records` is non-empty and every value is finite.
///
/// # Errors
///
/// Returns [`MlError::EmptySamples`]/[`MlError::EmptyFeatures`] if `records`
/// has no rows or columns, or [`MlError::NonFiniteFeature`] if any value is
/// `NaN` or infinite.
pub fn validate_features(records: ArrayView2<'_, f64>) -> Result<()> {
    validate_feature_shape(records)?;

    if let Some(((row, column), _)) = records.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(MlError::NonFiniteFeature { row, column });
    }

    Ok(())
}

/// Validates that `records` is non-empty and every value is non-infinite (`NaN` allowed).
///
/// # Errors
///
/// Returns [`MlError::EmptySamples`]/[`MlError::EmptyFeatures`] if `records`
/// has no rows or columns, or [`MlError::InfiniteFeature`] if any value is
/// infinite.
pub fn validate_features_allow_nan(records: ArrayView2<'_, f64>) -> Result<()> {
    validate_feature_shape(records)?;

    if let Some(((row, column), _)) = records
        .indexed_iter()
        .find(|(_, value)| value.is_infinite())
    {
        return Err(MlError::InfiniteFeature { row, column });
    }

    Ok(())
}

/// Validates that `records` has at least one row and one column.
///
/// # Errors
///
/// Returns [`MlError::EmptySamples`] if `records` has no rows, or
/// [`MlError::EmptyFeatures`] if it has no columns.
pub fn validate_feature_shape(records: ArrayView2<'_, f64>) -> Result<()> {
    if records.nrows() == 0 {
        return Err(MlError::EmptySamples);
    }
    if records.ncols() == 0 {
        return Err(MlError::EmptyFeatures);
    }

    Ok(())
}

/// Validates that `actual` matches the `expected` feature count.
///
/// # Errors
///
/// Returns [`MlError::MismatchedFeatureCount`] if `actual` differs from
/// `expected`.
pub fn validate_feature_count(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(MlError::MismatchedFeatureCount { expected, actual });
    }
    Ok(())
}
