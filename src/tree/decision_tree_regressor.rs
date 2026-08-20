// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView2};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    tree::common::{
        GrowthLimits, Node, build_tree, feature_importances, target_mean, target_variance,
        validate_min_samples_leaf, validate_min_samples_split,
    },
};

/// Configures a CART-style decision-tree regressor.
///
/// Splits minimize sample-size-weighted variance among the midpoints between
/// consecutive distinct sorted feature values.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecisionTreeRegressor {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }
}

impl DecisionTreeRegressor {
    /// Creates a decision-tree regressor with unrestricted depth.
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

    /// Grows a decision tree that minimizes weighted variance at every split.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid minimum split or leaf sample count,
    /// when features are empty or non-finite, or when a target is non-finite.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedDecisionTreeRegressor> {
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_features(dataset.records())?;
        for (index, &target) in dataset.targets().iter().enumerate() {
            if !target.is_finite() {
                return Err(MlError::NonFiniteActualTarget { index });
            }
        }

        let targets = dataset.targets().to_owned();
        let weights = Array1::from_elem(dataset.n_samples(), 1.0);
        let impurity = |rows: &[usize]| target_variance(&targets, rows);
        let make_leaf = |rows: &[usize]| target_mean(&targets, rows);
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
            weights.view(),
            &impurity,
            &make_leaf,
            &mut feature_sampler,
        );

        Ok(FittedDecisionTreeRegressor {
            root,
            n_features: dataset.n_features(),
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        })
    }
}

impl Fit<&Dataset<f64>, ()> for DecisionTreeRegressor {
    type Fitted = FittedDecisionTreeRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// A decision tree learned by [`DecisionTreeRegressor`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedDecisionTreeRegressor {
    root: Node<f64>,
    n_features: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl FittedDecisionTreeRegressor {
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

    /// Predicts continuous targets for a feature matrix.
    ///
    /// Each prediction is the mean training target observed in the leaf a
    /// query row falls into.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;
        Ok(Array1::from_iter(
            records
                .rows()
                .into_iter()
                .map(|row| *self.root.leaf_for(row)),
        ))
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedDecisionTreeRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}
