// `ArrayView2` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array2, ArrayView2};

use super::{
    FittedMinMaxScaler, FittedPolynomialFeatures, FittedSimpleImputer, FittedStandardScaler,
    MinMaxScaler, PolynomialFeatures, SimpleImputer, StandardScaler,
};
use crate::core::{Result, Transform, validate_feature_shape, validate_features};

/// An unfitted feature transformer that can be stored in a [`Pipeline`].
///
/// Implement this trait to make an application-defined transformer usable as a
/// pipeline step. Implementations must learn only from `records` and return all
/// learned state in the fitted transformer.
pub trait TransformerEstimator: Send + Sync {
    /// Learns transformation parameters from a feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is invalid or fitting cannot produce a
    /// valid transformer.
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>>;
}

/// A fitted feature transformer that can be stored in a [`FittedPipeline`].
pub trait FittedTransformer: Send + Sync {
    /// Applies the learned transformation to a feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is incompatible with the fitted state or
    /// transformation cannot produce valid output.
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>>;
}

/// An ordered collection of unfitted feature transformers.
///
/// During fitting, each estimator is fitted against the output of the previous
/// fitted step. This ensures every step learns exclusively from the matrix
/// supplied to [`Pipeline::fit`]. An empty pipeline acts as an identity
/// transformer.
#[derive(Default)]
pub struct Pipeline {
    steps: Vec<Box<dyn TransformerEstimator>>,
}

impl Pipeline {
    /// Creates an empty identity pipeline.
    #[must_use]
    pub const fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Appends an estimator to the pipeline.
    #[must_use]
    pub fn then<Estimator>(mut self, estimator: Estimator) -> Self
    where
        Estimator: TransformerEstimator + 'static,
    {
        self.steps.push(Box::new(estimator));
        self
    }

    /// Returns the number of configured transformation steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether the pipeline has no transformation steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Fits every transformation step in order.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is invalid or any step fails to fit or
    /// transform its input.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedPipeline> {
        if self.steps.is_empty() {
            validate_features(records)?;
        } else {
            validate_feature_shape(records)?;
        }

        let mut current = records.to_owned();
        let mut fitted_steps = Vec::with_capacity(self.steps.len());
        for estimator in &self.steps {
            let fitted = estimator.fit(current.view())?;
            current = fitted.transform(current.view())?;
            fitted_steps.push(fitted);
        }

        Ok(FittedPipeline {
            steps: fitted_steps,
        })
    }
}

/// The learned state of a [`Pipeline`].
pub struct FittedPipeline {
    steps: Vec<Box<dyn FittedTransformer>>,
}

impl FittedPipeline {
    /// Returns the number of fitted transformation steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether the fitted pipeline has no transformation steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Applies each fitted transformation in order.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` is invalid or incompatible with any
    /// fitted step.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if self.steps.is_empty() {
            validate_features(records)?;
        } else {
            validate_feature_shape(records)?;
        }

        let mut current = records.to_owned();
        for transformer in &self.steps {
            current = transformer.transform(current.view())?;
        }
        Ok(current)
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedPipeline {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

impl TransformerEstimator for StandardScaler {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedStandardScaler {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}

impl TransformerEstimator for MinMaxScaler {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedMinMaxScaler {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}

impl TransformerEstimator for SimpleImputer {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedSimpleImputer {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}

impl TransformerEstimator for PolynomialFeatures {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedPolynomialFeatures {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}
