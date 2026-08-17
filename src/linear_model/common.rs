// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView2};

use crate::core::{MlError, Result, validate_feature_count, validate_features};

pub(super) fn predict_linear(
    coefficients: &Array1<f64>,
    intercept: f64,
    records: ArrayView2<'_, f64>,
) -> Result<Array1<f64>> {
    validate_features(records)?;
    validate_feature_count(records.ncols(), coefficients.len())?;
    let mut predictions = records.dot(coefficients);
    predictions += intercept;
    if let Some((index, _prediction)) = predictions
        .iter()
        .enumerate()
        .find(|(_index, prediction)| !prediction.is_finite())
    {
        return Err(MlError::NonFinitePrediction { index });
    }
    Ok(predictions)
}
