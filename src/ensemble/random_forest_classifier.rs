// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};
use rand::{RngExt, SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;

use crate::{
    core::{Dataset, Fit, Predict, Result, validate_feature_count, validate_features},
    ensemble::common::{
        MaxFeatures, max_features_count, validate_max_features, validate_n_estimators,
    },
    tree::{
        GrowthLimits, Node, build_tree, feature_importances, gini_impurity, leaf_probabilities,
        validate_min_samples_leaf, validate_min_samples_split,
    },
};

const DEFAULT_N_ESTIMATORS: usize = 100;
const DEFAULT_SEED: u64 = 42;

/// Configures a bagged ensemble of CART-style decision-tree classifiers.
///
/// Every tree is trained on an independent bootstrap sample (drawn with
/// replacement) of the training rows, and every split considers a freshly
/// drawn random subset of features. Predictions average per-tree leaf class
/// probabilities. Classes are stored in sorted order, making predictions
/// deterministic for a given seed even when training rows are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomForestClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    seed: u64,
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self {
            n_estimators: DEFAULT_N_ESTIMATORS,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            seed: DEFAULT_SEED,
        }
    }
}

impl RandomForestClassifier {
    /// Creates a random-forest classifier with the default configuration:
    /// 100 trees of unrestricted depth, each splitting on a random subset of
    /// `sqrt(n_features)` features.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            n_estimators: DEFAULT_N_ESTIMATORS,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
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

    /// Grows a bootstrap-aggregated forest of Gini-minimizing decision trees.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid estimator count, minimum split or leaf
    /// sample count, or maximum feature count, or when features are empty or
    /// non-finite.
    pub fn fit<Label>(
        &self,
        dataset: &Dataset<Label>,
    ) -> Result<FittedRandomForestClassifier<Label>>
    where
        Label: Clone + Ord,
    {
        validate_n_estimators(self.n_estimators)?;
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_max_features(self.max_features)?;
        validate_features(dataset.records())?;

        let classes = sorted_classes(dataset);
        let n_classes = classes.len();
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();
        let encoded: Array1<usize> = Array1::from_iter(
            dataset
                .targets()
                .iter()
                .map(|label| classes.binary_search(label).unwrap_or(0)),
        );

        let impurity = |rows: &[usize]| gini_impurity(&encoded, n_classes, rows);
        let make_leaf = |rows: &[usize]| leaf_probabilities(&encoded, n_classes, rows);
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
                &impurity,
                &make_leaf,
                &mut feature_sampler,
            ));
        }

        Ok(FittedRandomForestClassifier {
            trees,
            classes,
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

impl<Label> Fit<&Dataset<Label>, ()> for RandomForestClassifier
where
    Label: Clone + Ord,
{
    type Fitted = FittedRandomForestClassifier<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// A forest of decision trees learned by [`RandomForestClassifier`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedRandomForestClassifier<Label> {
    trees: Vec<Node<Array1<f64>>>,
    classes: Vec<Label>,
    n_features: usize,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    seed: u64,
}

impl<Label> FittedRandomForestClassifier<Label> {
    /// Returns classes in probability-column order.
    #[must_use]
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// Returns the number of learned classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

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

    /// Predicts normalized class probabilities for every sample.
    ///
    /// Each row averages the class-frequency distribution reported by every
    /// tree's leaf for that sample. Column order matches [`Self::classes`].
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn predict_probabilities(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        #[allow(clippy::cast_precision_loss)]
        let n_estimators = self.n_estimators as f64;
        let mut probabilities = Array2::zeros((records.nrows(), self.n_classes()));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            let mut averaged = probabilities.row_mut(row_index);
            for tree in &self.trees {
                averaged += tree.leaf_for(row);
            }
            averaged.mapv_inplace(|total| total / n_estimators);
        }
        Ok(probabilities)
    }
}

impl<Label> FittedRandomForestClassifier<Label>
where
    Label: Clone,
{
    /// Predicts the class with the greatest averaged probability.
    ///
    /// Ties are resolved in favor of the first sorted class.
    ///
    /// # Errors
    ///
    /// Returns the same feature errors as [`Self::predict_probabilities`].
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<Label>> {
        let probabilities = self.predict_probabilities(records)?;
        Ok(Array1::from_iter(probabilities.rows().into_iter().map(
            |row| {
                let mut best_class = 0;
                for class_index in 1..self.n_classes() {
                    if row[class_index] > row[best_class] {
                        best_class = class_index;
                    }
                }
                self.classes[best_class].clone()
            },
        )))
    }
}

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedRandomForestClassifier<Label>
where
    Label: Clone,
{
    type Output = Array1<Label>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

fn sorted_classes<Label>(dataset: &Dataset<Label>) -> Vec<Label>
where
    Label: Clone + Ord,
{
    let mut classes = dataset.targets().to_vec();
    classes.sort_unstable();
    classes.dedup();
    classes
}
