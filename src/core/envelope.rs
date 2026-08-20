use crate::core::{MlError, Result};

/// Format version written by [`ModelEnvelope::new`] in this crate version.
///
/// Bump this constant whenever the envelope's own shape changes in a way
/// that is not backward compatible. It is independent of any individual
/// estimator's fields, which `serde` already versions implicitly through
/// struct shape.
pub const ENVELOPE_VERSION: u32 = 1;

/// A version-tagged wrapper around a serialized model.
///
/// Serializing a fitted model directly (as every fitted model in this crate
/// already supports under the `serde` feature) is enough for same-version
/// round trips. `ModelEnvelope` adds an explicit format version alongside
/// the model so that code reading serialized data can detect a version it
/// does not support before trying to use a possibly-incompatible model,
/// rather than failing with a confusing downstream error. See the
/// `model_envelope` example for a complete round trip.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelEnvelope<T> {
    version: u32,
    model: T,
}

impl<T> ModelEnvelope<T> {
    /// Wraps `model` with the current envelope format version.
    #[must_use]
    pub const fn new(model: T) -> Self {
        Self {
            version: ENVELOPE_VERSION,
            model,
        }
    }

    /// Returns the format version recorded in this envelope.
    ///
    /// For an envelope constructed with [`Self::new`] this is always
    /// [`ENVELOPE_VERSION`]; a deserialized envelope may carry a different
    /// value if it was written by a different crate version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the wrapped model without checking its envelope version.
    #[must_use]
    pub const fn model(&self) -> &T {
        &self.model
    }

    /// Consumes the envelope, returning the wrapped model only if its
    /// version matches [`ENVELOPE_VERSION`].
    ///
    /// # Errors
    ///
    /// Returns [`MlError::UnsupportedEnvelopeVersion`] when the envelope's
    /// version does not match the version this crate version supports.
    pub fn into_model(self) -> Result<T> {
        if self.version != ENVELOPE_VERSION {
            return Err(MlError::UnsupportedEnvelopeVersion {
                found: self.version,
                supported: ENVELOPE_VERSION,
            });
        }
        Ok(self.model)
    }
}
