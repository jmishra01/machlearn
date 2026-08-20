use crate::core::{MlError, Result};

/// Controls how many features are considered at each split of each tree in a
/// random forest.
///
/// A fresh random subset of this size is drawn independently at every split,
/// not once per tree, matching standard random-forest practice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MaxFeatures {
    /// Every feature is a candidate at every split.
    All,
    /// `floor(sqrt(n_features))` features, at least one.
    Sqrt,
    /// `floor(log2(n_features))` features, at least one.
    Log2,
    /// A fixed number of features, clamped to the available feature count.
    Fixed(usize),
}

pub(super) fn validate_max_features(max_features: MaxFeatures) -> Result<()> {
    if matches!(max_features, MaxFeatures::Fixed(0)) {
        return Err(MlError::InvalidMaxFeatures(0));
    }
    Ok(())
}

pub(super) fn validate_n_estimators(n_estimators: usize) -> Result<()> {
    if n_estimators == 0 {
        return Err(MlError::InvalidEstimatorCount(n_estimators));
    }
    Ok(())
}

/// Resolves [`MaxFeatures`] to a concrete count, clamped to `[1, n_features]`.
pub(super) fn max_features_count(max_features: MaxFeatures, n_features: usize) -> usize {
    let requested = match max_features {
        MaxFeatures::All => n_features,
        MaxFeatures::Sqrt => {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation
            )]
            let count = (n_features as f64).sqrt().floor() as usize;
            count
        }
        MaxFeatures::Log2 => {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation
            )]
            let count = (n_features as f64).log2().floor() as usize;
            count
        }
        MaxFeatures::Fixed(count) => count,
    };
    requested.clamp(1, n_features)
}
