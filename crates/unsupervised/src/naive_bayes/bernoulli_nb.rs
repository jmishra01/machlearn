// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::naive_bayes::common::validate_alpha;
use machlearn_core::core::{
    Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features,
};

const DEFAULT_ALPHA: f64 = 1.0;
const DEFAULT_BINARIZE: Option<f64> = Some(0.0);

/// Configures Bernoulli Naive Bayes classification over binary (presence or
/// absence) features.
///
/// Every feature is first binarized against `binarize` (a value strictly
/// greater than the threshold becomes `1`, everything else becomes `0`;
/// pass `None` to skip binarization when features are already `0`/`1`).
/// Each class's per-feature likelihood is the Laplace-smoothed fraction of
/// that class's rows in which the binarized feature is present:
/// `(present_count(class, feature) + alpha) / (class_count + 2 * alpha)`,
/// and an absent feature contributes its complement, `1 - p`, unlike
/// multinomial Naive Bayes, which only scores present features. Classes are
/// stored in sorted order, making predictions deterministic even when
/// training rows are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BernoulliNaiveBayes {
    alpha: f64,
    fit_prior: bool,
    binarize: Option<f64>,
}

impl Default for BernoulliNaiveBayes {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            fit_prior: true,
            binarize: DEFAULT_BINARIZE,
        }
    }
}

impl BernoulliNaiveBayes {
    /// Creates a Bernoulli Naive Bayes classifier with additive smoothing
    /// `alpha = 1.0` (Laplace smoothing), priors learned from the training
    /// class frequencies, and features binarized at a threshold of zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            fit_prior: true,
            binarize: DEFAULT_BINARIZE,
        }
    }

    /// Sets the additive (Lidstone) smoothing applied to every feature
    /// count.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha` is negative, NaN, or infinite.
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    /// Sets whether class priors are learned from the training class
    /// frequencies (`true`, the default) or assumed uniform (`false`).
    #[must_use]
    pub const fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Sets the binarization threshold: a feature value strictly greater
    /// than `threshold` becomes `1`, everything else becomes `0`. Pass
    /// `None` to skip binarization when features are already `0`/`1`.
    #[must_use]
    pub const fn with_binarize(mut self, threshold: Option<f64>) -> Self {
        self.binarize = threshold;
        self
    }

    /// Returns the configured smoothing factor.
    #[must_use]
    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    /// Returns whether priors are learned from training class frequencies.
    #[must_use]
    pub const fn fit_prior(self) -> bool {
        self.fit_prior
    }

    /// Returns the configured binarization threshold.
    #[must_use]
    pub const fn binarize(self) -> Option<f64> {
        self.binarize
    }

    /// Fits per-class feature and prior log-probabilities.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative smoothing factor, or when features
    /// are empty or non-finite.
    pub fn fit<Label>(&self, dataset: &Dataset<Label>) -> Result<FittedBernoulliNaiveBayes<Label>>
    where
        Label: Clone + Ord,
    {
        validate_alpha(self.alpha)?;
        validate_features(dataset.records())?;

        let binarized = binarize(dataset.records(), self.binarize);

        let classes = sorted_classes(dataset);
        let n_classes = classes.len();
        let n_features = dataset.n_features();
        #[allow(clippy::cast_precision_loss)]
        let total_samples = dataset.n_samples() as f64;
        #[allow(clippy::cast_precision_loss)]
        let uniform_log_prior = -(n_classes as f64).ln();

        let mut class_log_prior = Array1::zeros(n_classes);
        let mut feature_log_prob = Array2::zeros((n_classes, n_features));

        for (class_index, class) in classes.iter().enumerate() {
            let rows: Vec<usize> = dataset
                .targets()
                .iter()
                .enumerate()
                .filter_map(|(row_index, label)| (label == class).then_some(row_index))
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let class_count = rows.len() as f64;
            let present_counts: Array1<f64> = binarized.select(Axis(0), &rows).sum_axis(Axis(0));
            let denominator = class_count + 2.0 * self.alpha;
            for feature_index in 0..n_features {
                feature_log_prob[[class_index, feature_index]] =
                    ((present_counts[feature_index] + self.alpha) / denominator).ln();
            }

            class_log_prior[class_index] = if self.fit_prior {
                (class_count / total_samples).ln()
            } else {
                uniform_log_prior
            };
        }

        Ok(FittedBernoulliNaiveBayes {
            classes,
            class_log_prior,
            feature_log_prob,
            alpha: self.alpha,
            fit_prior: self.fit_prior,
            binarize: self.binarize,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for BernoulliNaiveBayes
where
    Label: Clone + Ord,
{
    type Fitted = FittedBernoulliNaiveBayes<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Parameters learned by [`BernoulliNaiveBayes`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedBernoulliNaiveBayes<Label> {
    classes: Vec<Label>,
    class_log_prior: Array1<f64>,
    feature_log_prob: Array2<f64>,
    alpha: f64,
    fit_prior: bool,
    binarize: Option<f64>,
}

impl<Label> FittedBernoulliNaiveBayes<Label> {
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
    pub fn n_features(&self) -> usize {
        self.feature_log_prob.ncols()
    }

    /// Returns the configured smoothing factor.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns whether priors were learned from training class frequencies.
    #[must_use]
    pub const fn fit_prior(&self) -> bool {
        self.fit_prior
    }

    /// Returns the configured binarization threshold.
    #[must_use]
    pub const fn binarize(&self) -> Option<f64> {
        self.binarize
    }

    /// Returns the class priors, in class order.
    #[must_use]
    pub fn class_priors(&self) -> Array1<f64> {
        self.class_log_prior.mapv(f64::exp)
    }

    /// Returns a matrix containing one per-feature presence log-probability
    /// row per class.
    #[must_use]
    pub const fn feature_log_prob(&self) -> &Array2<f64> {
        &self.feature_log_prob
    }

    /// Computes one joint log-likelihood score per class and sample.
    ///
    /// Each score sums the class log prior with every (binarized) feature's
    /// contribution: its presence log-probability when present, or the
    /// complementary absence log-probability when absent.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite score.
    pub fn decision_function(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;

        let binarized = binarize(records, self.binarize);
        let absence_log_prob = self.feature_log_prob.mapv(|log_p| (-log_p.exp()).ln_1p());
        let presence_delta = &self.feature_log_prob - &absence_log_prob;
        let absence_total = absence_log_prob.sum_axis(Axis(1));

        let mut scores = binarized.dot(&presence_delta.t());
        for mut row in scores.rows_mut() {
            row += &absence_total;
            row += &self.class_log_prior;
        }
        if let Some((index, _score)) = scores
            .rows()
            .into_iter()
            .enumerate()
            .find(|(_index, row)| !row.iter().all(|value| value.is_finite()))
        {
            return Err(MlError::NonFinitePrediction { index });
        }
        Ok(scores)
    }

    /// Predicts normalized class probabilities for every sample.
    ///
    /// Joint log-likelihoods are normalized in log space, so every row
    /// remains finite and sums to one even for extreme scores. Column order
    /// matches [`Self::classes`].
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
    pub fn predict_probabilities(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut probabilities = self.decision_function(records)?;
        for mut row in probabilities.rows_mut() {
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|value| (value - maximum).exp());
            let normalizer = row.sum();
            row.mapv_inplace(|value| value / normalizer);
        }
        Ok(probabilities)
    }
}

impl<Label> FittedBernoulliNaiveBayes<Label>
where
    Label: Clone,
{
    /// Predicts the class with the greatest joint log-likelihood.
    ///
    /// Ties are resolved in favor of the first sorted class.
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<Label>> {
        let scores = self.decision_function(records)?;
        Ok(Array1::from_iter(scores.rows().into_iter().map(|row| {
            let mut best_class = 0;
            for class_index in 1..self.n_classes() {
                if row[class_index] > row[best_class] {
                    best_class = class_index;
                }
            }
            self.classes[best_class].clone()
        })))
    }
}

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedBernoulliNaiveBayes<Label>
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

fn binarize(records: ArrayView2<'_, f64>, threshold: Option<f64>) -> Array2<f64> {
    threshold.map_or_else(
        || records.to_owned(),
        |threshold| records.mapv(|value| if value > threshold { 1.0 } else { 0.0 }),
    )
}
