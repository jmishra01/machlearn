mod dataset;
#[cfg(feature = "serde")]
mod envelope;
mod error;
mod traits;
mod validation;

pub use dataset::Dataset;
#[cfg(feature = "serde")]
pub use envelope::{ENVELOPE_VERSION, ModelEnvelope};
pub use error::{MlError, Result};
pub use traits::{Fit, Predict, Transform};
pub(crate) use validation::{
    validate_feature_count, validate_feature_shape, validate_features, validate_features_allow_nan,
};
