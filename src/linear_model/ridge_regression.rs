// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView2, Axis};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result},
    linear_model::{LinearRegression, common::predict_linear},
    solver::solve_ridge,
};

/// Configures L2-regularized linear regression.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RidgeRegression {
    alpha: f64,
    fit_intercept: bool,
}

impl Default for RidgeRegression {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }
}

impl RidgeRegression {
    /// Creates a ridge regressor with regularization strength `alpha`.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha` is negative, NaN, or infinite.
    pub fn new(alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        Ok(Self {
            alpha,
            fit_intercept: true,
        })
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

    /// Fits ridge regression using an augmented least-squares system.
    ///
    /// When fitting an intercept, features and targets are centered first so
    /// the intercept is not regularized. With `alpha = 0`, fitting delegates to
    /// ordinary least squares and preserves its validation behavior.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid regularization, non-finite targets or
    /// derived values, or an OLS rank/sample failure when `alpha` is zero.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedRidgeRegression> {
        validate_alpha(self.alpha)?;
        if is_zero(self.alpha) {
            let ordinary = LinearRegression::new()
                .with_intercept(self.fit_intercept)
                .fit(dataset)?;
            return Ok(FittedRidgeRegression {
                coefficients: ordinary.coefficients().clone(),
                intercept: ordinary.intercept(),
                alpha: self.alpha,
            });
        }

        if self.fit_intercept {
            let (centered_records, centered_targets, feature_means, target_mean) = center(dataset)?;
            let coefficients =
                solve_ridge(centered_records.view(), centered_targets.view(), self.alpha)?;
            let intercept = target_mean - feature_means.dot(&coefficients);
            if !intercept.is_finite() {
                return Err(MlError::NonFiniteSolverOutput {
                    index: coefficients.len(),
                });
            }
            Ok(FittedRidgeRegression {
                coefficients,
                intercept,
                alpha: self.alpha,
            })
        } else {
            let coefficients = solve_ridge(dataset.records(), dataset.targets(), self.alpha)?;
            Ok(FittedRidgeRegression {
                coefficients,
                intercept: 0.0,
                alpha: self.alpha,
            })
        }
    }
}

impl Fit<&Dataset<f64>, ()> for RidgeRegression {
    type Fitted = FittedRidgeRegression;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Coefficients learned by [`RidgeRegression`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedRidgeRegression {
    coefficients: Array1<f64>,
    intercept: f64,
    alpha: f64,
}

impl FittedRidgeRegression {
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

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.coefficients.len()
    }

    /// Predicts continuous targets for a feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite prediction.
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        predict_linear(&self.coefficients, self.intercept, records)
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedRidgeRegression {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

fn validate_alpha(alpha: f64) -> Result<()> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(MlError::InvalidRegularization(alpha));
    }
    Ok(())
}

#[allow(clippy::float_cmp)]
fn is_zero(value: f64) -> bool {
    value == 0.0
}

pub(super) type CenteredData = (ndarray::Array2<f64>, Array1<f64>, Array1<f64>, f64);

pub(super) fn center(dataset: &Dataset<f64>) -> Result<CenteredData> {
    for (index, &target) in dataset.targets().iter().enumerate() {
        if !target.is_finite() {
            return Err(MlError::NonFiniteActualTarget { index });
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let sample_count = dataset.n_samples() as f64;
    let feature_means = Array1::from_iter(dataset.records().axis_iter(Axis(1)).map(|feature| {
        feature
            .iter()
            .map(|value| value / sample_count)
            .sum::<f64>()
    }));
    let target_mean = dataset
        .targets()
        .iter()
        .map(|target| target / sample_count)
        .sum();
    let mut centered_records = dataset.records().to_owned();
    for mut row in centered_records.rows_mut() {
        row -= &feature_means;
    }
    let centered_targets = dataset.targets().mapv(|target| target - target_mean);
    Ok((
        centered_records,
        centered_targets,
        feature_means,
        target_mean,
    ))
}
