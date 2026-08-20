// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView2};
use rand::{RngExt, SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    ensemble::common::{
        MaxFeatures, max_features_count, validate_max_features, validate_n_estimators,
    },
    tree::{
        GrowthLimits, Node, build_tree, feature_importances, target_mean, target_variance,
        validate_min_samples_leaf, validate_min_samples_split,
    },
};

const DEFAULT_N_ESTIMATORS: usize = 100;
const DEFAULT_SEED: u64 = 42;

/// Configures a bagged ensemble of CART-style decision-tree regressors.
///
/// Every tree is trained on an independent bootstrap sample (drawn with
/// replacement) of the training rows, and every split considers a freshly
/// drawn random subset of features. Predictions average per-tree leaf means.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomForestRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    seed: u64,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: DEFAULT_N_ESTIMATORS,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            seed: DEFAULT_SEED,
        }
    }
}

impl RandomForestRegressor {
    /// Creates a random-forest regressor with the default configuration: 100
    /// trees of unrestricted depth, each considering every feature at every
    /// split.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            n_estimators: DEFAULT_N_ESTIMATORS,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            seed: DEFAULT_SEED,
        }
    }

    /// Sets the number of trees in the forest.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_estimators` is zero.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Result<Self> {
        validate_n_estimators(n_estimators)?;
        self.n_estimators = n_estimators;
        Ok(self)
    }

    /// Limits how many splits may occur along any root-to-leaf path of every
    /// tree.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum number of samples a node must have to be split.
    ///
    /// # Errors
    ///
    /// Returns an error when `min_samples_split` is less than two.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Result<Self> {
        validate_min_samples_split(min_samples_split)?;
        self.min_samples_split = min_samples_split;
        Ok(self)
    }

    /// Sets the minimum number of samples every leaf must retain.
    ///
    /// # Errors
    ///
    /// Returns an error when `min_samples_leaf` is zero.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Result<Self> {
        validate_min_samples_leaf(min_samples_leaf)?;
        self.min_samples_leaf = min_samples_leaf;
        Ok(self)
    }

    /// Sets how many features are randomly considered at every split.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_features` is `Fixed(0)`.
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Result<Self> {
        validate_max_features(max_features)?;
        self.max_features = max_features;
        Ok(self)
    }

    /// Sets the deterministic seed used for bootstrap sampling and feature
    /// selection.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the configured number of trees.
    #[must_use]
    pub const fn n_estimators(self) -> usize {
        self.n_estimators
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

    /// Returns the configured per-split feature-sampling strategy.
    #[must_use]
    pub const fn max_features(self) -> MaxFeatures {
        self.max_features
    }

    /// Returns the configured random seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Grows a bootstrap-aggregated forest of variance-minimizing decision
    /// trees.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid estimator count, minimum split or leaf
    /// sample count, or maximum feature count, when features are empty or
    /// non-finite, or when a target is non-finite.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedRandomForestRegressor> {
        validate_n_estimators(self.n_estimators)?;
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_max_features(self.max_features)?;
        validate_features(dataset.records())?;
        for (index, &target) in dataset.targets().iter().enumerate() {
            if !target.is_finite() {
                return Err(MlError::NonFiniteActualTarget { index });
            }
        }

        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();
        let targets = dataset.targets().to_owned();
        let weights = Array1::from_elem(n_samples, 1.0);
        let impurity = |rows: &[usize]| target_variance(&targets, rows);
        let make_leaf = |rows: &[usize]| target_mean(&targets, rows);
        let limits = GrowthLimits {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };
        let feature_count = max_features_count(self.max_features, n_features);

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _tree_index in 0..self.n_estimators {
            let bootstrap_rows: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let mut feature_sampler = |available: usize| {
                index::sample(&mut rng, available, feature_count.min(available)).into_vec()
            };
            trees.push(build_tree(
                dataset.records(),
                bootstrap_rows,
                0,
                &limits,
                weights.view(),
                &impurity,
                &make_leaf,
                &mut feature_sampler,
            ));
        }

        Ok(FittedRandomForestRegressor {
            trees,
            n_features,
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            seed: self.seed,
        })
    }
}

impl Fit<&Dataset<f64>, ()> for RandomForestRegressor {
    type Fitted = FittedRandomForestRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// A forest of decision trees learned by [`RandomForestRegressor`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedRandomForestRegressor {
    trees: Vec<Node<f64>>,
    n_features: usize,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    seed: u64,
}

impl FittedRandomForestRegressor {
    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of trees in the forest.
    #[must_use]
    pub const fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    /// Returns the configured depth limit.
    #[must_use]
    pub const fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Returns the configured minimum split sample count.
    #[must_use]
    pub const fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Returns the configured minimum leaf sample count.
    #[must_use]
    pub const fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Returns the configured per-split feature-sampling strategy.
    #[must_use]
    pub const fn max_features(&self) -> MaxFeatures {
        self.max_features
    }

    /// Returns the random seed used to grow the forest.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the mean-decrease-in-impurity importance of every feature,
    /// averaged across trees and normalized to sum to one.
    ///
    /// Each tree's importances are normalized before averaging, so every
    /// tree contributes equally regardless of how many splits it made.
    #[must_use]
    pub fn feature_importances(&self) -> Array1<f64> {
        #[allow(clippy::cast_precision_loss)]
        let n_estimators = self.n_estimators as f64;
        let mut averaged = Array1::zeros(self.n_features);
        for tree in &self.trees {
            averaged += &feature_importances(tree, self.n_features);
        }
        averaged / n_estimators
    }

    /// Predicts continuous targets for a feature matrix.
    ///
    /// Each prediction averages the leaf mean reported by every tree for
    /// that sample.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        #[allow(clippy::cast_precision_loss)]
        let n_estimators = self.n_estimators as f64;
        Ok(Array1::from_iter(records.rows().into_iter().map(|row| {
            let total: f64 = self.trees.iter().map(|tree| *tree.leaf_for(row)).sum();
            total / n_estimators
        })))
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedRandomForestRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}
