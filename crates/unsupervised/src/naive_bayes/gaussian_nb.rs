// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView2, Axis};

use machlearn_core::core::{
    Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features,
};

const MINIMUM_VARIANCE: f64 = 1.0e-12;
const DEFAULT_VAR_SMOOTHING: f64 = 1.0e-9;

/// Configures Gaussian Naive Bayes classification.
///
/// Each class's feature distribution is modeled as an independent Gaussian
/// fitted from the training data. Classes are stored in sorted order, making
/// predictions deterministic even when training rows are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianNaiveBayes {
    var_smoothing: f64,
}

impl Default for GaussianNaiveBayes {
    fn default() -> Self {
        Self {
            var_smoothing: DEFAULT_VAR_SMOOTHING,
        }
    }
}

impl GaussianNaiveBayes {
    /// Creates a Gaussian Naive Bayes classifier with the default variance
    /// smoothing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            var_smoothing: DEFAULT_VAR_SMOOTHING,
        }
    }

    /// Sets the variance-smoothing factor.
    ///
    /// This fraction of the largest per-feature variance observed across the
    /// whole training set is added to every class's per-feature variance,
    /// preventing zero-variance features from producing infinite densities.
    ///
    /// # Errors
    ///
    /// Returns an error when `var_smoothing` is negative, NaN, or infinite.
    pub fn with_var_smoothing(mut self, var_smoothing: f64) -> Result<Self> {
        validate_var_smoothing(var_smoothing)?;
        self.var_smoothing = var_smoothing;
        Ok(self)
    }

    /// Returns the configured variance-smoothing factor.
    #[must_use]
    pub const fn var_smoothing(self) -> f64 {
        self.var_smoothing
    }

    /// Fits per-class feature means, variances, and priors.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid variance smoothing or when features are
    /// empty or non-finite.
    pub fn fit<Label>(&self, dataset: &Dataset<Label>) -> Result<FittedGaussianNaiveBayes<Label>>
    where
        Label: Clone + Ord,
    {
        validate_var_smoothing(self.var_smoothing)?;
        validate_features(dataset.records())?;

        let classes = sorted_classes(dataset);
        let n_classes = classes.len();
        let n_features = dataset.n_features();
        #[allow(clippy::cast_precision_loss)]
        let total_samples = dataset.n_samples() as f64;
        let epsilon = self.var_smoothing * feature_variance_max(dataset.records());

        let mut class_log_priors = Array1::zeros(n_classes);
        let mut means = Array2::zeros((n_classes, n_features));
        let mut variances = Array2::zeros((n_classes, n_features));

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

            for (feature_index, column) in class_records.axis_iter(Axis(1)).enumerate() {
                let mean = column.sum() / class_count;
                let variance = column
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / class_count
                    + epsilon;
                means[[class_index, feature_index]] = mean;
                variances[[class_index, feature_index]] = variance.max(MINIMUM_VARIANCE);
            }
            class_log_priors[class_index] = (class_count / total_samples).ln();
        }

        Ok(FittedGaussianNaiveBayes {
            classes,
            class_log_priors,
            means,
            variances,
            var_smoothing: self.var_smoothing,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for GaussianNaiveBayes
where
    Label: Clone + Ord,
{
    type Fitted = FittedGaussianNaiveBayes<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Parameters learned by [`GaussianNaiveBayes`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedGaussianNaiveBayes<Label> {
    classes: Vec<Label>,
    class_log_priors: Array1<f64>,
    means: Array2<f64>,
    variances: Array2<f64>,
    var_smoothing: f64,
}

impl<Label> FittedGaussianNaiveBayes<Label> {
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
        self.means.ncols()
    }

    /// Returns the configured variance-smoothing factor.
    #[must_use]
    pub const fn var_smoothing(&self) -> f64 {
        self.var_smoothing
    }

    /// Returns the empirical class priors, in class order.
    #[must_use]
    pub fn class_priors(&self) -> Array1<f64> {
        self.class_log_priors.mapv(f64::exp)
    }

    /// Returns a matrix containing one per-feature mean row per class.
    #[must_use]
    pub const fn means(&self) -> &Array2<f64> {
        &self.means
    }

    /// Returns a matrix containing one per-feature variance row per class,
    /// including the fitted variance-smoothing term.
    #[must_use]
    pub const fn variances(&self) -> &Array2<f64> {
        &self.variances
    }

    /// Computes one joint log-likelihood score per class and sample.
    ///
    /// Each score sums the class log prior with the log density of every
    /// feature under its class-conditional Gaussian.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite score.
    pub fn decision_function(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;

        let mut scores = Array2::zeros((records.nrows(), self.n_classes()));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            for class_index in 0..self.n_classes() {
                let mut score = self.class_log_priors[class_index];
                for feature_index in 0..self.n_features() {
                    let mean = self.means[[class_index, feature_index]];
                    let variance = self.variances[[class_index, feature_index]];
                    let deviation = row[feature_index] - mean;
                    score -= 0.5 * ((2.0 * PI * variance).ln() + deviation * deviation / variance);
                }
                if !score.is_finite() {
                    return Err(MlError::NonFinitePrediction { index: row_index });
                }
                scores[[row_index, class_index]] = score;
            }
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

impl<Label> FittedGaussianNaiveBayes<Label>
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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedGaussianNaiveBayes<Label>
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

fn feature_variance_max(records: ArrayView2<'_, f64>) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n_samples = records.nrows() as f64;
    records
        .axis_iter(Axis(1))
        .map(|column| {
            let mean = column.sum() / n_samples;
            column
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / n_samples
        })
        .fold(0.0_f64, f64::max)
}

fn validate_var_smoothing(var_smoothing: f64) -> Result<()> {
    if !var_smoothing.is_finite() || var_smoothing < 0.0 {
        return Err(MlError::InvalidVarianceSmoothing(var_smoothing));
    }
    Ok(())
}
