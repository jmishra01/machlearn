// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView2};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    ensemble::common::validate_n_estimators,
    tree::{
        DecisionTreeRegressor, FittedDecisionTreeRegressor, validate_min_samples_leaf,
        validate_min_samples_split,
    },
};

const DEFAULT_N_ESTIMATORS: usize = 100;
const DEFAULT_LEARNING_RATE: f64 = 0.1;
const DEFAULT_MAX_DEPTH: Option<usize> = Some(3);

/// Configures gradient-boosted decision-tree regression.
///
/// Each boosting round fits a shallow [`crate::DecisionTreeRegressor`] to
/// the residual between the current ensemble prediction and the training
/// targets (the negative gradient of squared error), then adds that tree's
/// predictions to the ensemble, scaled by `learning_rate`. Unlike
/// [`crate::RandomForestRegressor`], trees are fitted sequentially, each
/// correcting the errors of the ones before it, rather than independently
/// on bootstrap samples.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self {
            n_estimators: DEFAULT_N_ESTIMATORS,
            learning_rate: DEFAULT_LEARNING_RATE,
            max_depth: DEFAULT_MAX_DEPTH,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }
}

impl GradientBoostingRegressor {
    /// Creates a gradient-boosting regressor with 100 depth-3 trees and a
    /// learning rate of `0.1`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            n_estimators: DEFAULT_N_ESTIMATORS,
            learning_rate: DEFAULT_LEARNING_RATE,
            max_depth: DEFAULT_MAX_DEPTH,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    /// Sets the number of boosting rounds (trees).
    ///
    /// # Errors
    ///
    /// Returns an error when `n_estimators` is zero.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Result<Self> {
        validate_n_estimators(n_estimators)?;
        self.n_estimators = n_estimators;
        Ok(self)
    }

    /// Sets the shrinkage applied to every tree's contribution to the
    /// ensemble.
    ///
    /// # Errors
    ///
    /// Returns an error when `learning_rate` is non-positive, NaN, or
    /// infinite.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(self)
    }

    /// Limits how many splits may occur along any root-to-leaf path of every
    /// tree.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum number of samples a node must have to be split, in
    /// every tree.
    ///
    /// # Errors
    ///
    /// Returns an error when `min_samples_split` is less than two.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Result<Self> {
        validate_min_samples_split(min_samples_split)?;
        self.min_samples_split = min_samples_split;
        Ok(self)
    }

    /// Sets the minimum number of samples every leaf must retain, in every
    /// tree.
    ///
    /// # Errors
    ///
    /// Returns an error when `min_samples_leaf` is zero.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Result<Self> {
        validate_min_samples_leaf(min_samples_leaf)?;
        self.min_samples_leaf = min_samples_leaf;
        Ok(self)
    }

    /// Returns the configured number of boosting rounds.
    #[must_use]
    pub const fn n_estimators(self) -> usize {
        self.n_estimators
    }

    /// Returns the configured learning rate.
    #[must_use]
    pub const fn learning_rate(self) -> f64 {
        self.learning_rate
    }

    /// Returns the configured depth limit.
    #[must_use]
    pub const fn max_depth(self) -> Option<usize> {
        self.max_depth
    }

    /// Returns the configured minimum split sample count.
    #[must_use]
    pub const fn min_samples_split(self) -> usize {
        self.min_samples_split
    }

    /// Returns the configured minimum leaf sample count.
    #[must_use]
    pub const fn min_samples_leaf(self) -> usize {
        self.min_samples_leaf
    }

    /// Fits a gradient-boosted ensemble of decision trees.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid estimator count, learning rate,
    /// minimum split, or minimum leaf sample count, when features are empty
    /// or non-finite, or when a target is non-finite.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedGradientBoostingRegressor> {
        validate_n_estimators(self.n_estimators)?;
        validate_learning_rate(self.learning_rate)?;
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_features(dataset.records())?;
        for (index, &target) in dataset.targets().iter().enumerate() {
            if !target.is_finite() {
                return Err(MlError::NonFiniteActualTarget { index });
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let sample_count = dataset.n_samples() as f64;
        let initial_prediction = dataset.targets().sum() / sample_count;

        let tree_estimator = DecisionTreeRegressor::new()
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)?
            .with_min_samples_leaf(self.min_samples_leaf)?;

        let mut predictions = Array1::from_elem(dataset.n_samples(), initial_prediction);
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _round in 0..self.n_estimators {
            let residuals = dataset.targets().to_owned() - &predictions;
            let residual_dataset = Dataset::new(dataset.records().to_owned(), residuals)?;
            let tree = tree_estimator.fit(&residual_dataset)?;
            predictions.scaled_add(self.learning_rate, &tree.predict(dataset.records())?);
            trees.push(tree);
        }

        Ok(FittedGradientBoostingRegressor {
            trees,
            initial_prediction,
            learning_rate: self.learning_rate,
            n_features: dataset.n_features(),
        })
    }
}

impl Fit<&Dataset<f64>, ()> for GradientBoostingRegressor {
    type Fitted = FittedGradientBoostingRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// An ensemble of decision trees learned by [`GradientBoostingRegressor`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedGradientBoostingRegressor {
    trees: Vec<FittedDecisionTreeRegressor>,
    initial_prediction: f64,
    learning_rate: f64,
    n_features: usize,
}

impl FittedGradientBoostingRegressor {
    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of trees in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Returns the configured learning rate.
    #[must_use]
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the constant prediction the ensemble starts from before any
    /// boosting round is applied.
    #[must_use]
    pub const fn initial_prediction(&self) -> f64 {
        self.initial_prediction
    }

    /// Predicts continuous targets for a feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite prediction.
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let mut predictions = Array1::from_elem(records.nrows(), self.initial_prediction);
        for tree in &self.trees {
            predictions.scaled_add(self.learning_rate, &tree.predict(records)?);
        }
        if let Some((index, _prediction)) = predictions
            .iter()
            .enumerate()
            .find(|(_index, prediction)| !prediction.is_finite())
        {
            return Err(MlError::NonFinitePrediction { index });
        }
        Ok(predictions)
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedGradientBoostingRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

pub(super) fn validate_learning_rate(learning_rate: f64) -> Result<()> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(MlError::InvalidLearningRate(learning_rate));
    }
    Ok(())
}
