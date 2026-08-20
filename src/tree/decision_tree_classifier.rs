// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    core::{Dataset, Fit, Predict, Result, validate_feature_count, validate_features},
    tree::common::{
        GrowthLimits, Node, build_tree, feature_importances, gini_impurity, leaf_probabilities,
        validate_min_samples_leaf, validate_min_samples_split,
    },
};

/// Configures a CART-style decision-tree classifier.
///
/// Splits minimize sample-size-weighted Gini impurity among the midpoints
/// between consecutive distinct sorted feature values. Classes are stored in
/// sorted order, making predictions deterministic even when training rows
/// are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecisionTreeClassifier {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }
}

impl DecisionTreeClassifier {
    /// Creates a decision-tree classifier with unrestricted depth.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    /// Limits how many splits may occur along any root-to-leaf path.
    ///
    /// `None` grows the tree until every leaf is pure or a sample-count
    /// constraint stops splitting.
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

    /// Grows a decision tree that minimizes Gini impurity at every split.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid minimum split or leaf sample count, or
    /// when features are empty or non-finite.
    pub fn fit<Label>(
        &self,
        dataset: &Dataset<Label>,
    ) -> Result<FittedDecisionTreeClassifier<Label>>
    where
        Label: Clone + Ord,
    {
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_features(dataset.records())?;

        let weights = Array1::from_elem(dataset.n_samples(), 1.0);
        self.fit_weighted(dataset, weights.view())
    }

    /// Grows a decision tree exactly as [`Self::fit`], but weights each
    /// training row's contribution to every split's impurity score and to
    /// its leaf's class distribution by `weights`, rather than treating
    /// every row equally.
    ///
    /// Used internally by ensembles (such as [`crate::AdaBoostClassifier`])
    /// that must re-fit a tree against per-round sample weights without
    /// resampling the training data.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::fit`].
    pub(crate) fn fit_weighted<Label>(
        &self,
        dataset: &Dataset<Label>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<FittedDecisionTreeClassifier<Label>>
    where
        Label: Clone + Ord,
    {
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_features(dataset.records())?;

        let classes = sorted_classes(dataset);
        let n_classes = classes.len();
        let encoded: Array1<usize> = Array1::from_iter(
            dataset
                .targets()
                .iter()
                .map(|label| classes.binary_search(label).unwrap_or(0)),
        );

        let impurity = |rows: &[usize]| gini_impurity(&encoded, n_classes, weights, rows);
        let make_leaf = |rows: &[usize]| leaf_probabilities(&encoded, n_classes, weights, rows);
        let limits = GrowthLimits {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };
        let rows: Vec<usize> = (0..dataset.n_samples()).collect();
        let mut feature_sampler = |n_features: usize| (0..n_features).collect::<Vec<_>>();
        let root = build_tree(
            dataset.records(),
            rows,
            0,
            &limits,
            weights,
            &impurity,
            &make_leaf,
            &mut feature_sampler,
        );

        Ok(FittedDecisionTreeClassifier {
            root,
            classes,
            n_features: dataset.n_features(),
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for DecisionTreeClassifier
where
    Label: Clone + Ord,
{
    type Fitted = FittedDecisionTreeClassifier<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// A decision tree learned by [`DecisionTreeClassifier`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedDecisionTreeClassifier<Label> {
    root: Node<Array1<f64>>,
    classes: Vec<Label>,
    n_features: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl<Label> FittedDecisionTreeClassifier<Label> {
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

    /// Returns the mean-decrease-in-impurity importance of every feature,
    /// normalized to sum to one.
    ///
    /// A feature that was never split on has an importance of zero. When the
    /// tree is a single leaf (no splits occurred), every importance is zero.
    #[must_use]
    pub fn feature_importances(&self) -> Array1<f64> {
        feature_importances(&self.root, self.n_features)
    }

    /// Predicts normalized class probabilities for every sample.
    ///
    /// Each row is the class-frequency distribution observed among the
    /// training samples in the leaf a query row falls into. Column order
    /// matches [`Self::classes`].
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn predict_probabilities(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let mut probabilities = Array2::zeros((records.nrows(), self.n_classes()));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            probabilities
                .row_mut(row_index)
                .assign(self.root.leaf_for(row));
        }
        Ok(probabilities)
    }
}

impl<Label> FittedDecisionTreeClassifier<Label>
where
    Label: Clone,
{
    /// Predicts the class with the greatest probability in its leaf.
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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedDecisionTreeClassifier<Label>
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
