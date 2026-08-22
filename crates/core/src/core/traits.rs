use crate::core::Result;

/// Fits an estimator to features and targets.
///
/// Implementations should keep configuration in the estimator and return an
/// immutable fitted model. This makes it impossible to call prediction APIs on
/// a model that has not been fitted.
pub trait Fit<X, Y> {
    /// Model produced by fitting the estimator.
    type Fitted;

    /// Fits the estimator to `features` and `targets`.
    ///
    /// # Errors
    ///
    /// Returns an error when the inputs are incompatible with the estimator or
    /// fitting cannot produce a valid model.
    fn fit(&self, features: X, targets: Y) -> Result<Self::Fitted>;
}

/// Generates predictions from a fitted model.
pub trait Predict<X> {
    /// Prediction container produced by the model.
    type Output;

    /// Predicts outputs for `features`.
    ///
    /// # Errors
    ///
    /// Returns an error when the features are incompatible with the fitted
    /// model or prediction cannot produce a valid output.
    fn predict(&self, features: X) -> Result<Self::Output>;
}

/// Transforms input data using fitted preprocessing state.
pub trait Transform<X> {
    /// Container produced by the transformation.
    type Output;

    /// Applies the fitted transformation to `input`.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is incompatible with the fitted
    /// transformer or transformation cannot produce a valid output.
    fn transform(&self, input: X) -> Result<Self::Output>;
}
