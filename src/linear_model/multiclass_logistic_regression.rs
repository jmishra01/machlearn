// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    linear_model::{
        common::log_sigmoid,
        logistic_regression::{fit_parameters, validate_alpha},
    },
};

/// Configures L2-regularized multiclass logistic regression.
///
/// One binary logistic model is fitted for each class against all remaining
/// classes. Learned classes and probability columns are sorted using [`Ord`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MulticlassLogisticRegression {
    alpha: f64,
    fit_intercept: bool,
}

impl Default for MulticlassLogisticRegression {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }
}

impl MulticlassLogisticRegression {
    /// Creates a one-vs-rest classifier with unit L2 regularization.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }

    /// Sets the L2 regularization strength for every class model.
    ///
    /// Intercepts are never regularized. Zero disables regularization and
    /// therefore requires a full-rank design for every one-vs-rest fit.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha` is negative, NaN, or infinite.
    pub fn with_regularization(mut self, alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    /// Enables or disables intercept fitting for every class model.
    #[must_use]
    pub const fn with_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }

    /// Returns the L2 regularization strength.
    #[must_use]
    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    /// Returns whether unpenalized intercepts will be fitted.
    #[must_use]
    pub const fn fit_intercept(self) -> bool {
        self.fit_intercept
    }

    /// Fits one binary logistic model per observed class.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid regularization, fewer than three target
    /// classes, a rank-deficient unregularized design, a numerical failure, or
    /// failure of any class model to converge.
    pub fn fit<Label>(
        &self,
        dataset: &Dataset<Label>,
    ) -> Result<FittedMulticlassLogisticRegression<Label>>
    where
        Label: Clone + Ord,
    {
        validate_alpha(self.alpha)?;
        let classes = sorted_multiclass_targets(dataset)?;
        let mut coefficients = Array2::zeros((classes.len(), dataset.n_features()));
        let mut intercepts = Array1::zeros(classes.len());

        for (class_index, class) in classes.iter().enumerate() {
            let encoded = Array1::from_iter(
                dataset
                    .targets()
                    .iter()
                    .map(|label| usize::from(label == class)),
            );
            let parameters =
                fit_parameters(dataset.records(), &encoded, self.fit_intercept, self.alpha)?;
            let coefficient_start = usize::from(self.fit_intercept);
            if self.fit_intercept {
                intercepts[class_index] = parameters[0];
            }
            for feature_index in 0..dataset.n_features() {
                coefficients[[class_index, feature_index]] =
                    parameters[feature_index + coefficient_start];
            }
        }

        Ok(FittedMulticlassLogisticRegression {
            coefficients,
            intercepts,
            alpha: self.alpha,
            classes,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for MulticlassLogisticRegression
where
    Label: Clone + Ord,
{
    type Fitted = FittedMulticlassLogisticRegression<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// One-vs-rest parameters learned by [`MulticlassLogisticRegression`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedMulticlassLogisticRegression<Label> {
    coefficients: Array2<f64>,
    intercepts: Array1<f64>,
    alpha: f64,
    classes: Vec<Label>,
}

impl<Label> FittedMulticlassLogisticRegression<Label> {
    /// Returns a matrix containing one coefficient row per class.
    #[must_use]
    pub const fn coefficients(&self) -> &Array2<f64> {
        &self.coefficients
    }

    /// Returns one fitted intercept per class.
    #[must_use]
    pub const fn intercepts(&self) -> &Array1<f64> {
        &self.intercepts
    }

    /// Returns the regularization strength used during fitting.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

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

    /// Computes one one-vs-rest decision score per class and sample.
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
    /// Individual one-vs-rest probabilities are normalized in log space, so
    /// every row remains finite and sums to one even for extreme scores.
    /// Column order matches [`Self::classes`].
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
    pub fn predict_probabilities(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let scores = self.decision_function(records)?;
        let mut probabilities = scores.mapv(log_sigmoid);
        for mut row in probabilities.rows_mut() {
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|value| (value - maximum).exp());
            let normalizer = row.sum();
            row.mapv_inplace(|value| value / normalizer);
        }
        Ok(probabilities)
    }
}

impl<Label> FittedMulticlassLogisticRegression<Label>
where
    Label: Clone,
{
    /// Predicts the class with the greatest normalized probability.
    ///
    /// Probability ties are resolved in favor of the first sorted class.
    ///
    /// # Errors
    ///
    /// Returns the same feature and numerical errors as [`Self::decision_function`].
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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedMulticlassLogisticRegression<Label>
where
    Label: Clone,
{
    type Output = Array1<Label>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

fn sorted_multiclass_targets<Label>(dataset: &Dataset<Label>) -> Result<Vec<Label>>
where
    Label: Clone + Ord,
{
    let mut classes = dataset.targets().to_vec();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() < 3 {
        return Err(MlError::ExpectedMulticlassTargets {
            class_count: classes.len(),
        });
    }
    Ok(classes)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::linear_model::common::log_sigmoid;

    #[test]
    fn log_sigmoid_is_stable_for_extreme_scores() {
        assert_abs_diff_eq!(log_sigmoid(0.0), -2.0_f64.ln());
        assert_abs_diff_eq!(log_sigmoid(1_000.0), 0.0);
        assert_abs_diff_eq!(log_sigmoid(-1_000.0), -1_000.0);
    }
}
