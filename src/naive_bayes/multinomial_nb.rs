// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    naive_bayes::common::{validate_alpha, validate_non_negative_features},
};

const DEFAULT_ALPHA: f64 = 1.0;

/// Configures multinomial Naive Bayes classification over non-negative
/// count-like features (such as word frequencies).
///
/// Each class's per-feature likelihood is the Lidstone-smoothed relative
/// frequency of that feature among the total feature count observed for the
/// class: `(count(class, feature) + alpha) / (count(class) + alpha *
/// n_features)`. Classes are stored in sorted order, making predictions
/// deterministic even when training rows are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultinomialNaiveBayes {
    alpha: f64,
    fit_prior: bool,
}

impl Default for MultinomialNaiveBayes {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            fit_prior: true,
        }
    }
}

impl MultinomialNaiveBayes {
    /// Creates a multinomial Naive Bayes classifier with additive smoothing
    /// `alpha = 1.0` (Laplace smoothing) and priors learned from the
    /// training class frequencies.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            fit_prior: true,
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

    /// Fits per-class feature and prior log-probabilities.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative smoothing factor, when features are
    /// empty or non-finite, or when a feature value is negative.
    pub fn fit<Label>(&self, dataset: &Dataset<Label>) -> Result<FittedMultinomialNaiveBayes<Label>>
    where
        Label: Clone + Ord,
    {
        validate_alpha(self.alpha)?;
        validate_features(dataset.records())?;
        validate_non_negative_features(dataset.records())?;

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
            let class_records = dataset.records().select(Axis(0), &rows);

            let feature_counts: Array1<f64> = class_records.sum_axis(Axis(0));
            let total_feature_count = feature_counts.sum();
            #[allow(clippy::cast_precision_loss)]
            let denominator = total_feature_count + self.alpha * n_features as f64;
            for feature_index in 0..n_features {
                feature_log_prob[[class_index, feature_index]] =
                    ((feature_counts[feature_index] + self.alpha) / denominator).ln();
            }

            class_log_prior[class_index] = if self.fit_prior {
                (class_count / total_samples).ln()
            } else {
                uniform_log_prior
            };
        }

        Ok(FittedMultinomialNaiveBayes {
            classes,
            class_log_prior,
            feature_log_prob,
            alpha: self.alpha,
            fit_prior: self.fit_prior,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for MultinomialNaiveBayes
where
    Label: Clone + Ord,
{
    type Fitted = FittedMultinomialNaiveBayes<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Parameters learned by [`MultinomialNaiveBayes`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedMultinomialNaiveBayes<Label> {
    classes: Vec<Label>,
    class_log_prior: Array1<f64>,
    feature_log_prob: Array2<f64>,
    alpha: f64,
    fit_prior: bool,
}

impl<Label> FittedMultinomialNaiveBayes<Label> {
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

    /// Returns the class priors, in class order.
    #[must_use]
    pub fn class_priors(&self) -> Array1<f64> {
        self.class_log_prior.mapv(f64::exp)
    }

    /// Returns a matrix containing one per-feature log-probability row per
    /// class.
    #[must_use]
    pub const fn feature_log_prob(&self) -> &Array2<f64> {
        &self.feature_log_prob
    }

    /// Computes one joint log-likelihood score per class and sample.
    ///
    /// Each score sums the class log prior with every feature count scaled
    /// by its class-conditional log-probability.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, negative, have
    /// the wrong column count, or produce a non-finite score.
    pub fn decision_function(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;
        validate_non_negative_features(records)?;

        let mut scores = records.dot(&self.feature_log_prob.t());
        for mut row in scores.rows_mut() {
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

impl<Label> FittedMultinomialNaiveBayes<Label>
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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedMultinomialNaiveBayes<Label>
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
