// `ArrayView2` is a lightweight view descriptor; accepting it by value keeps
// call sites consistent with the rest of the ndarray ecosystem.
#![allow(clippy::needless_pass_by_value)]

use ndarray::ArrayView2;

use crate::core::{MlError, Result};

pub(crate) fn validate_features(records: ArrayView2<'_, f64>) -> Result<()> {
    if records.nrows() == 0 {
        return Err(MlError::EmptySamples);
    }
    if records.ncols() == 0 {
        return Err(MlError::EmptyFeatures);
    }

    if let Some(((row, column), _)) = records.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(MlError::NonFiniteFeature { row, column });
    }

    Ok(())
}

pub(crate) fn validate_feature_count(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(MlError::MismatchedFeatureCount { expected, actual });
    }
    Ok(())
}
