// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    ensemble::{
        common::validate_n_estimators, gradient_boosting_regressor::validate_learning_rate,
    },
    tree::{
        DecisionTreeRegressor, FittedDecisionTreeRegressor, validate_min_samples_leaf,
        validate_min_samples_split,
    },
};

const DEFAULT_N_ESTIMATORS: usize = 100;
const DEFAULT_LEARNING_RATE: f64 = 0.1;
const DEFAULT_MAX_DEPTH: Option<usize> = Some(3);

// A leaf whose samples all carry near-identical predicted probabilities has
// a Hessian sum near zero; a Newton step there would divide by (near) zero.
const MINIMUM_LEAF_HESSIAN: f64 = 1e-12;

/// Configures gradient-boosted decision-tree binary classification.
///
/// Each boosting round fits a shallow [`crate::DecisionTreeRegressor`] to
/// the residual between the current ensemble's predicted probability and
/// the training labels (the negative gradient of log loss). Every leaf's
/// prediction is then replaced by a single Newton-Raphson step computed
/// from the log-loss gradient and Hessian of the samples that fall into it,
/// matching the standard `TreeBoost` refinement used for non-squared-error
/// losses. That corrected tree output is added to the ensemble's decision
/// score, scaled by `learning_rate`. Classes are stored in sorted order:
/// the first is negative and the second positive, matching
/// [`crate::LogisticRegression`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl Default for GradientBoostingClassifier {
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

impl GradientBoostingClassifier {
    /// Creates a gradient-boosting classifier with 100 depth-3 trees and a
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
    /// ensemble's decision score.
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

    /// Fits a gradient-boosted ensemble of decision trees to minimize log
    /// loss.
    ///
    /// Labels may be any cloneable ordered type. Exactly two distinct
    /// classes must occur in the training targets.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid estimator count, learning rate,
    /// minimum split, or minimum leaf sample count, for a target collection
    /// that does not contain exactly two classes, or when features are
    /// empty or non-finite.
    pub fn fit<Label>(
        &self,
        dataset: &Dataset<Label>,
    ) -> Result<FittedGradientBoostingClassifier<Label>>
    where
        Label: Clone + Ord,
    {
        validate_n_estimators(self.n_estimators)?;
        validate_learning_rate(self.learning_rate)?;
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_features(dataset.records())?;

        let classes = sorted_binary_classes(dataset)?;
        let encoded = Array1::from_iter(
            dataset
                .targets()
                .iter()
                .map(|label| if label == &classes[1] { 1.0 } else { 0.0 }),
        );

        #[allow(clippy::cast_precision_loss)]
        let sample_count = dataset.n_samples() as f64;
        let positive_fraction = encoded.sum() / sample_count;
        let initial_score = (positive_fraction / (1.0 - positive_fraction)).ln();

        let tree_estimator = DecisionTreeRegressor::new()
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)?
            .with_min_samples_leaf(self.min_samples_leaf)?;

        let mut scores = Array1::from_elem(dataset.n_samples(), initial_score);
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _round in 0..self.n_estimators {
            let probabilities = scores.mapv(sigmoid);
            let residuals = &encoded - &probabilities;
            let hessians = probabilities.mapv(|probability| probability * (1.0 - probability));
            let boosted_tree =
                BoostedTree::fit(&tree_estimator, dataset.records(), &residuals, &hessians)?;
            scores.scaled_add(
                self.learning_rate,
                &boosted_tree.predict(dataset.records())?,
            );
            trees.push(boosted_tree);
        }

        Ok(FittedGradientBoostingClassifier {
            trees,
            initial_score,
            learning_rate: self.learning_rate,
            n_features: dataset.n_features(),
            classes,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for GradientBoostingClassifier
where
    Label: Clone + Ord,
{
    type Fitted = FittedGradientBoostingClassifier<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// A regression tree together with the Newton-corrected value each of its
/// leaves contributes to the ensemble's decision score.
///
/// A leaf's raw regression output (the mean residual of the samples that
/// reached it) is only the *right direction* to move the score for
/// non-squared-error losses; the leaf constants below are recomputed as a
/// single Newton-Raphson step so the boosting round matches the standard
/// `TreeBoost` algorithm. Leaves are identified by the bit pattern of the
/// tree's raw prediction, which is exactly reproducible for any input that
/// reaches that leaf.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct BoostedTree {
    tree: FittedDecisionTreeRegressor,
    leaf_values: Vec<(u64, f64)>,
}

impl BoostedTree {
    fn fit(
        tree_estimator: &DecisionTreeRegressor,
        records: ArrayView2<'_, f64>,
        residuals: &Array1<f64>,
        hessians: &Array1<f64>,
    ) -> Result<Self> {
        let residual_dataset = Dataset::new(records.to_owned(), residuals.clone())?;
        let tree = tree_estimator.fit(&residual_dataset)?;
        let raw_predictions = tree.predict(records)?;

        let mut numerators: Vec<(u64, f64)> = Vec::new();
        let mut denominators: Vec<(u64, f64)> = Vec::new();
        for (index, raw_prediction) in raw_predictions.iter().enumerate() {
            let key = raw_prediction.to_bits();
            accumulate(&mut numerators, key, residuals[index]);
            accumulate(&mut denominators, key, hessians[index]);
        }

        let leaf_values = numerators
            .into_iter()
            .map(|(key, numerator)| {
                let denominator = denominators
                    .iter()
                    .find(|(candidate, _)| *candidate == key)
                    .map_or(0.0, |(_, value)| *value);
                let corrected_value = if denominator > MINIMUM_LEAF_HESSIAN {
                    numerator / denominator
                } else {
                    0.0
                };
                (key, corrected_value)
            })
            .collect();

        Ok(Self { tree, leaf_values })
    }

    fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let raw_predictions = self.tree.predict(records)?;
        Ok(raw_predictions.mapv(|raw_prediction| {
            let key = raw_prediction.to_bits();
            self.leaf_values
                .iter()
                .find(|(candidate, _)| *candidate == key)
                .map_or(raw_prediction, |(_, corrected_value)| *corrected_value)
        }))
    }
}

fn accumulate(pairs: &mut Vec<(u64, f64)>, key: u64, value: f64) {
    if let Some(entry) = pairs.iter_mut().find(|(candidate, _)| *candidate == key) {
        entry.1 += value;
    } else {
        pairs.push((key, value));
    }
}

/// An ensemble of decision trees learned by [`GradientBoostingClassifier`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedGradientBoostingClassifier<Label> {
    trees: Vec<BoostedTree>,
    initial_score: f64,
    learning_rate: f64,
    n_features: usize,
    classes: Vec<Label>,
}

impl<Label> FittedGradientBoostingClassifier<Label> {
    /// Returns the two classes in negative-then-positive order.
    #[must_use]
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// Returns the class represented by probability column zero.
    #[must_use]
    pub fn negative_class(&self) -> &Label {
        &self.classes[0]
    }

    /// Returns the class represented by probability column one.
    #[must_use]
    pub fn positive_class(&self) -> &Label {
        &self.classes[1]
    }

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

    /// Computes the ensemble's decision score for every row.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite score.
    pub fn decision_function(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let mut scores = Array1::from_elem(records.nrows(), self.initial_score);
        for tree in &self.trees {
            scores.scaled_add(self.learning_rate, &tree.predict(records)?);
        }
        if let Some((index, _score)) = scores
            .iter()
            .enumerate()
            .find(|(_index, score)| !score.is_finite())
        {
            return Err(MlError::NonFinitePrediction { index });
        }
        Ok(scores)
    }

    /// Predicts positive-class probabilities for every row.
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
    pub fn predict_positive_probabilities(
        &self,
        records: ArrayView2<'_, f64>,
    ) -> Result<Array1<f64>> {
        Ok(self.decision_function(records)?.mapv(sigmoid))
    }

    /// Predicts one probability column per class.
    ///
    /// Column order matches [`Self::classes`].
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
    pub fn predict_probabilities(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let positive = self.predict_positive_probabilities(records)?;
        Ok(Array2::from_shape_fn(
            (positive.len(), 2),
            |(row, column)| {
                if column == 0 {
                    1.0 - positive[row]
                } else {
                    positive[row]
                }
            },
        ))
    }
}

impl<Label> FittedGradientBoostingClassifier<Label>
where
    Label: Clone,
{
    /// Predicts class labels using a positive-class threshold of one half.
    ///
    /// A probability equal to one half is assigned to the positive class.
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<Label>> {
        self.predict_positive_probabilities(records)
            .map(|probabilities| {
                probabilities.mapv(|probability| {
                    if probability >= 0.5 {
                        self.classes[1].clone()
                    } else {
                        self.classes[0].clone()
                    }
                })
            })
    }
}

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedGradientBoostingClassifier<Label>
where
    Label: Clone,
{
    type Output = Array1<Label>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

fn sorted_binary_classes<Label>(dataset: &Dataset<Label>) -> Result<Vec<Label>>
where
    Label: Clone + Ord,
{
    let mut classes = dataset.targets().to_vec();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() != 2 {
        return Err(MlError::ExpectedBinaryTargets {
            class_count: classes.len(),
        });
    }
    Ok(classes)
}

fn sigmoid(score: f64) -> f64 {
    if score >= 0.0 {
        1.0 / (1.0 + (-score).exp())
    } else {
        let exponential = score.exp();
        exponential / (1.0 + exponential)
    }
}
