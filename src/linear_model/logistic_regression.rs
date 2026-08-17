// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_features},
    linear_model::common::predict_linear,
    solver::solve_least_squares,
};

const MAX_ITERATIONS: usize = 100;
const CONVERGENCE_TOLERANCE: f64 = 1.0e-10;
const MINIMUM_WEIGHT: f64 = 1.0e-12;

/// Configures L2-regularized binary logistic regression.
///
/// Classes are stored in sorted order. The first class is treated as negative
/// and the second as positive, making class assignment deterministic even when
/// training rows are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogisticRegression {
    alpha: f64,
    fit_intercept: bool,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }
}

impl LogisticRegression {
    /// Creates a binary logistic classifier with unit L2 regularization.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }

    /// Sets the L2 regularization strength.
    ///
    /// The intercept is never regularized. A value of zero disables
    /// regularization and therefore requires a full-rank training design.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha` is negative, NaN, or infinite.
    pub fn with_regularization(mut self, alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    /// Enables or disables intercept fitting.
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

    /// Returns whether an unpenalized intercept will be fitted.
    #[must_use]
    pub const fn fit_intercept(self) -> bool {
        self.fit_intercept
    }

    /// Fits a binary classifier using iteratively reweighted least squares.
    ///
    /// Labels may be any cloneable ordered type. Exactly two distinct classes
    /// must occur in the training targets.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid regularization, a target collection that
    /// does not contain exactly two classes, a rank-deficient unregularized
    /// design, a numerical failure, or failure to converge.
    pub fn fit<Label>(&self, dataset: &Dataset<Label>) -> Result<FittedLogisticRegression<Label>>
    where
        Label: Clone + Ord,
    {
        validate_alpha(self.alpha)?;
        let classes = sorted_binary_classes(dataset)?;
        let encoded = Array1::from_iter(
            dataset
                .targets()
                .iter()
                .map(|label| usize::from(label == &classes[1])),
        );
        let parameters =
            fit_parameters(dataset.records(), &encoded, self.fit_intercept, self.alpha)?;
        let (intercept, coefficient_start) = if self.fit_intercept {
            (parameters[0], 1)
        } else {
            (0.0, 0)
        };
        let coefficients = Array1::from_iter(parameters.iter().skip(coefficient_start).copied());
        Ok(FittedLogisticRegression {
            coefficients,
            intercept,
            alpha: self.alpha,
            classes,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for LogisticRegression
where
    Label: Clone + Ord,
{
    type Fitted = FittedLogisticRegression<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Parameters and class labels learned by [`LogisticRegression`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedLogisticRegression<Label> {
    coefficients: Array1<f64>,
    intercept: f64,
    alpha: f64,
    classes: Vec<Label>,
}

impl<Label> FittedLogisticRegression<Label> {
    /// Returns one coefficient per input feature.
    #[must_use]
    pub const fn coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }

    /// Returns the fitted intercept, or zero when intercept fitting was disabled.
    #[must_use]
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Returns the regularization strength used during fitting.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

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
    pub fn n_features(&self) -> usize {
        self.coefficients.len()
    }

    /// Computes the linear decision score for every row.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite score.
    pub fn decision_function(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        predict_linear(&self.coefficients, self.intercept, records)
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

impl<Label> FittedLogisticRegression<Label>
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

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedLogisticRegression<Label>
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

fn fit_parameters(
    records: ArrayView2<'_, f64>,
    encoded: &Array1<usize>,
    fit_intercept: bool,
    alpha: f64,
) -> Result<Array1<f64>> {
    validate_features(records)?;

    let offset = usize::from(fit_intercept);
    let parameter_count = records.ncols() + offset;
    let penalty_rows = if alpha > 0.0 { records.ncols() } else { 0 };
    let mut parameters = Array1::zeros(parameter_count);
    if fit_intercept {
        #[allow(clippy::cast_precision_loss)]
        let positive_fraction = encoded.iter().sum::<usize>() as f64 / encoded.len() as f64;
        parameters[0] = (positive_fraction / (1.0 - positive_fraction)).ln();
    }

    for _iteration in 0..MAX_ITERATIONS {
        let mut design = Array2::zeros((records.nrows() + penalty_rows, parameter_count));
        let mut targets = Array1::zeros(records.nrows() + penalty_rows);

        for (row_index, row) in records.rows().into_iter().enumerate() {
            let mut score = if fit_intercept { parameters[0] } else { 0.0 };
            for (feature_index, &value) in row.iter().enumerate() {
                score += value * parameters[feature_index + offset];
            }
            if !score.is_finite() {
                return Err(MlError::NonFinitePrediction { index: row_index });
            }
            let probability = sigmoid(score);
            let weight = (probability * (1.0 - probability)).max(MINIMUM_WEIGHT);
            let square_root_weight = weight.sqrt();
            let target = if encoded[row_index] == 1 { 1.0 } else { 0.0 };
            let adjusted_target = score + (target - probability) / weight;

            if fit_intercept {
                design[[row_index, 0]] = square_root_weight;
            }
            for (feature_index, &value) in row.iter().enumerate() {
                design[[row_index, feature_index + offset]] = square_root_weight * value;
            }
            targets[row_index] = square_root_weight * adjusted_target;
        }

        if alpha > 0.0 {
            let penalty = alpha.sqrt();
            for feature_index in 0..records.ncols() {
                design[[records.nrows() + feature_index, feature_index + offset]] = penalty;
            }
        }

        let updated = solve_least_squares(design.view(), targets.view())?;
        let maximum_change = updated
            .iter()
            .zip(&parameters)
            .map(|(next, current)| (next - current).abs())
            .fold(0.0_f64, f64::max);
        let scale = updated
            .iter()
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        parameters = updated;
        if maximum_change <= CONVERGENCE_TOLERANCE * scale {
            return Ok(parameters);
        }
    }

    Err(MlError::OptimizationDidNotConverge {
        iterations: MAX_ITERATIONS,
    })
}

fn validate_alpha(alpha: f64) -> Result<()> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(MlError::InvalidRegularization(alpha));
    }
    Ok(())
}

fn sigmoid(score: f64) -> f64 {
    if score >= 0.0 {
        1.0 / (1.0 + (-score).exp())
    } else {
        let exponential = score.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::sigmoid;

    #[test]
    fn sigmoid_is_stable_for_extreme_scores() {
        assert_abs_diff_eq!(sigmoid(0.0), 0.5);
        assert_abs_diff_eq!(sigmoid(1_000.0), 1.0);
        assert_abs_diff_eq!(sigmoid(-1_000.0), 0.0);
    }
}
