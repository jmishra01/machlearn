// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    solver::solve_least_squares,
};

/// Configures linear discriminant analysis (LDA).
///
/// LDA models every class's features as a multivariate Gaussian sharing one
/// covariance matrix across all classes, estimated by pooling each class's
/// within-class scatter. Because the covariance is shared, the resulting
/// decision boundary between any two classes is linear, and predictions
/// reduce to a per-class linear discriminant score. Classes are stored in
/// sorted order, making predictions deterministic even when training rows
/// are reordered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearDiscriminantAnalysis;

impl LinearDiscriminantAnalysis {
    /// Creates a linear discriminant analysis classifier.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Fits per-class linear discriminant coefficients from a pooled,
    /// empirical class covariance.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty or non-finite, or when the
    /// pooled covariance is rank-deficient (for example, when there are
    /// fewer samples than features).
    pub fn fit<Label>(
        &self,
        dataset: &Dataset<Label>,
    ) -> Result<FittedLinearDiscriminantAnalysis<Label>>
    where
        Label: Clone + Ord,
    {
        validate_features(dataset.records())?;

        let classes = sorted_classes(dataset);
        let n_classes = classes.len();
        let n_features = dataset.n_features();
        let n_samples = dataset.n_samples();

        let encoded: Vec<usize> = dataset
            .targets()
            .iter()
            .map(|label| classes.binary_search(label).unwrap_or(0))
            .collect();

        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut class_counts = vec![0usize; n_classes];
        for (row_index, &class_index) in encoded.iter().enumerate() {
            class_means
                .row_mut(class_index)
                .scaled_add(1.0, &dataset.records().row(row_index));
            class_counts[class_index] += 1;
        }
        let mut class_log_priors = Array1::zeros(n_classes);
        #[allow(clippy::cast_precision_loss)]
        let sample_count = n_samples as f64;
        for (class_index, &count) in class_counts.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = count as f64;
            class_means
                .row_mut(class_index)
                .mapv_inplace(|sum| sum / count_f64);
            class_log_priors[class_index] = (count_f64 / sample_count).ln();
        }

        let mut covariance = Array2::zeros((n_features, n_features));
        for (row_index, &class_index) in encoded.iter().enumerate() {
            let deviation = &dataset.records().row(row_index) - &class_means.row(class_index);
            for (feature_i, &value_i) in deviation.iter().enumerate() {
                for (feature_j, &value_j) in deviation.iter().enumerate() {
                    covariance[[feature_i, feature_j]] += value_i * value_j;
                }
            }
        }
        covariance.mapv_inplace(|value| value / sample_count);

        let mut coefficients = Array2::zeros((n_classes, n_features));
        let mut intercepts = Array1::zeros(n_classes);
        for class_index in 0..n_classes {
            let mean = class_means.row(class_index);
            let weights = solve_least_squares(covariance.view(), mean)?;
            let bias = class_log_priors[class_index] - 0.5 * mean.dot(&weights);
            coefficients.row_mut(class_index).assign(&weights);
            intercepts[class_index] = bias;
        }

        Ok(FittedLinearDiscriminantAnalysis {
            classes,
            coefficients,
            intercepts,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for LinearDiscriminantAnalysis
where
    Label: Clone + Ord,
{
    type Fitted = FittedLinearDiscriminantAnalysis<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Discriminant coefficients learned by [`LinearDiscriminantAnalysis`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedLinearDiscriminantAnalysis<Label> {
    classes: Vec<Label>,
    coefficients: Array2<f64>,
    intercepts: Array1<f64>,
}

impl<Label> FittedLinearDiscriminantAnalysis<Label> {
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
        self.coefficients.ncols()
    }

    /// Returns a matrix containing one linear discriminant coefficient row
    /// per class.
    #[must_use]
    pub const fn coefficients(&self) -> &Array2<f64> {
        &self.coefficients
    }

    /// Returns one discriminant intercept per class.
    #[must_use]
    pub const fn intercepts(&self) -> &Array1<f64> {
        &self.intercepts
    }

    /// Computes one linear discriminant score per class and sample.
    ///
    /// Each score is proportional to the log posterior of that class under
    /// the shared-covariance Gaussian model, up to a class-independent
    /// additive constant.
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
                let score =
                    row.dot(&self.coefficients.row(class_index)) + self.intercepts[class_index];
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
    /// Discriminant scores are normalized in log space, so every row remains
    /// finite and sums to one even for extreme scores. Column order matches
    /// [`Self::classes`].
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

impl<Label> FittedLinearDiscriminantAnalysis<Label>
where
    Label: Clone,
{
    /// Predicts the class with the greatest discriminant score.
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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedLinearDiscriminantAnalysis<Label>
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
