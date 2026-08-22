// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};

use crate::{
    ensemble::{
        common::validate_n_estimators, gradient_boosting_regressor::validate_learning_rate,
    },
    tree::{
        DecisionTreeClassifier, FittedDecisionTreeClassifier, validate_min_samples_leaf,
        validate_min_samples_split,
    },
};
use machlearn_core::core::{
    Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features,
};

const DEFAULT_N_ESTIMATORS: usize = 50;
const DEFAULT_LEARNING_RATE: f64 = 1.0;
const DEFAULT_MAX_DEPTH: Option<usize> = Some(1);

/// Configures `AdaBoost` (discrete SAMME) binary classification over decision
/// stumps.
///
/// Each round fits a shallow [`crate::tree::DecisionTreeClassifier`] against the
/// current per-sample weights, computes its weighted error rate, and derives
/// a voting weight from it: `learning_rate * ln((1 - error) / error)`.
/// Misclassified rows' weights are then scaled by `exp(voting_weight)` and
/// every weight is renormalized, concentrating subsequent rounds on the
/// samples still being misclassified. Boosting stops early if a round's
/// weak learner is perfect (its lone vote decides the ensemble) or no
/// better than random guessing (the round is discarded and boosting halts
/// with whatever rounds already succeeded). Classes are stored in sorted
/// order: the first is negative and the second positive, matching
/// `LogisticRegression`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdaBoostClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl Default for AdaBoostClassifier {
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

impl AdaBoostClassifier {
    /// Creates an `AdaBoost` classifier with 50 decision-stump rounds and a
    /// learning rate of `1.0`.
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

    /// Sets the maximum number of boosting rounds.
    ///
    /// Boosting may stop earlier than this if a round's weak learner is
    /// perfect or no better than random guessing.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_estimators` is zero.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Result<Self> {
        validate_n_estimators(n_estimators)?;
        self.n_estimators = n_estimators;
        Ok(self)
    }

    /// Sets the shrinkage applied to every round's voting weight.
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
    /// weak learner.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum number of samples a node must have to be split, in
    /// every weak learner.
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
    /// weak learner.
    ///
    /// # Errors
    ///
    /// Returns an error when `min_samples_leaf` is zero.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Result<Self> {
        validate_min_samples_leaf(min_samples_leaf)?;
        self.min_samples_leaf = min_samples_leaf;
        Ok(self)
    }

    /// Returns the configured maximum number of boosting rounds.
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

    /// Fits an `AdaBoost` (discrete SAMME) ensemble of decision stumps.
    ///
    /// Labels may be any cloneable ordered type. Exactly two distinct
    /// classes must occur in the training targets.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid estimator count, learning rate,
    /// minimum split, or minimum leaf sample count, for a target collection
    /// that does not contain exactly two classes, when features are empty
    /// or non-finite, or when the very first weak learner performs no
    /// better than random guessing.
    pub fn fit<Label>(&self, dataset: &Dataset<Label>) -> Result<FittedAdaBoostClassifier<Label>>
    where
        Label: Clone + Ord,
    {
        validate_n_estimators(self.n_estimators)?;
        validate_learning_rate(self.learning_rate)?;
        validate_min_samples_split(self.min_samples_split)?;
        validate_min_samples_leaf(self.min_samples_leaf)?;
        validate_features(dataset.records())?;

        let classes = sorted_binary_classes(dataset)?;
        let n_samples = dataset.n_samples();
        let targets = dataset.targets();

        let stump_estimator = DecisionTreeClassifier::new()
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)?
            .with_min_samples_leaf(self.min_samples_leaf)?;

        #[allow(clippy::cast_precision_loss)]
        let mut sample_weight = Array1::from_elem(n_samples, 1.0 / n_samples as f64);
        let mut estimators: Vec<(FittedDecisionTreeClassifier<Label>, f64)> =
            Vec::with_capacity(self.n_estimators);

        for round in 0..self.n_estimators {
            for weight in &mut sample_weight {
                *weight = weight.max(f64::EPSILON);
            }

            let stump = stump_estimator.fit_weighted(dataset, sample_weight.view())?;
            let predictions = stump.predict(dataset.records())?;

            let mut weighted_error = 0.0;
            let mut weight_sum = 0.0;
            let mut incorrect = vec![false; n_samples];
            for index in 0..n_samples {
                weight_sum += sample_weight[index];
                if predictions[index] != targets[index] {
                    incorrect[index] = true;
                    weighted_error += sample_weight[index];
                }
            }
            let estimator_error = weighted_error / weight_sum;

            if estimator_error <= 0.0 {
                estimators.push((stump, 1.0));
                break;
            }
            if estimator_error >= 0.5 {
                if estimators.is_empty() {
                    return Err(MlError::WeakLearnerNoBetterThanRandom);
                }
                break;
            }

            let alpha = self.learning_rate * ((1.0 - estimator_error) / estimator_error).ln();
            estimators.push((stump, alpha));

            if round != self.n_estimators - 1 {
                for index in 0..n_samples {
                    if incorrect[index] {
                        sample_weight[index] *= alpha.exp();
                    }
                }
                let total: f64 = sample_weight.sum();
                if !total.is_finite() || total <= 0.0 {
                    break;
                }
                sample_weight.mapv_inplace(|weight| weight / total);
            }
        }

        Ok(FittedAdaBoostClassifier {
            estimators,
            n_features: dataset.n_features(),
            classes,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for AdaBoostClassifier
where
    Label: Clone + Ord,
{
    type Fitted = FittedAdaBoostClassifier<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// An ensemble of decision stumps learned by [`AdaBoostClassifier`].
///
/// Every stump is paired with the voting weight ("alpha") its round earned;
/// a stump that turned out perfect on a training round short-circuits the
/// ensemble to that single stump with a weight of one.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedAdaBoostClassifier<Label> {
    estimators: Vec<(FittedDecisionTreeClassifier<Label>, f64)>,
    n_features: usize,
    classes: Vec<Label>,
}

impl<Label> FittedAdaBoostClassifier<Label> {
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

    /// Returns the number of weak learners actually fit.
    ///
    /// May be smaller than the configured `n_estimators` when boosting
    /// stopped early.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }
}

impl<Label> FittedAdaBoostClassifier<Label>
where
    Label: Clone + PartialEq,
{
    /// Computes the ensemble's decision score for every row.
    ///
    /// Each weak learner casts a vote of `+1` for the positive class or
    /// `-1` for the negative class, scaled by its voting weight; the
    /// weighted sum is normalized by the total voting weight and doubled
    /// (matching the classic `AdaBoost` margin, whose halved value estimates
    /// the positive-class log-odds).
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite score.
    pub fn decision_function(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let alpha_total: f64 = self.estimators.iter().map(|(_, alpha)| alpha).sum();
        let mut scores = Array1::<f64>::zeros(records.nrows());
        for (stump, alpha) in &self.estimators {
            let predictions = stump.predict(records)?;
            for (index, prediction) in predictions.iter().enumerate() {
                let vote = if *prediction == self.classes[1] {
                    1.0
                } else {
                    -1.0
                };
                scores[index] += alpha * vote;
            }
        }
        scores.mapv_inplace(|score| 2.0 * score / alpha_total);

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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedAdaBoostClassifier<Label>
where
    Label: Clone + PartialEq,
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
