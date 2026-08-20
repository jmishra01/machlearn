// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::ArrayView2;

use crate::core::{MlError, Result};

/// Rejects any negative feature value.
///
/// Count-based Naive Bayes variants (multinomial, Bernoulli) model features
/// as non-negative frequencies or indicators; a negative value has no
/// meaningful likelihood under either model.
pub(super) fn validate_non_negative_features(records: ArrayView2<'_, f64>) -> Result<()> {
    if let Some(((row, column), _)) = records.indexed_iter().find(|&(_, &value)| value < 0.0) {
        return Err(MlError::NegativeFeature { row, column });
    }
    Ok(())
}

/// Rejects a negative, NaN, or infinite additive-smoothing factor.
pub(super) fn validate_alpha(alpha: f64) -> Result<()> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(MlError::InvalidAlpha(alpha));
    }
    Ok(())
}
