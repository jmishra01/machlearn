// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView2};

use crate::{
    core::{Dataset, Fit, Predict, Result},
    linear_model::{
        common::predict_linear,
        convergence::ConvergenceReport,
        coordinate_descent::fit_coordinate_descent,
        logistic_regression::{validate_alpha, validate_max_iterations, validate_tolerance},
    },
};

const DEFAULT_MAX_ITERATIONS: usize = 1000;
const DEFAULT_TOLERANCE: f64 = 1.0e-4;

/// Configures L1-regularized (Lasso) linear regression.
///
/// Minimizes `(1 / 2n) * ||y - Xw - b||^2 + alpha * ||w||_1` by coordinate
/// descent. The L1 penalty drives coefficients for uninformative features
/// to exactly zero, making Lasso useful for feature selection as well as
/// regression. Unlike [`crate::LinearRegression`] and
/// [`crate::RidgeRegression`], fitting does not require at least as many
/// samples as features.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LassoRegression {
    alpha: f64,
    fit_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
}

impl LassoRegression {
    /// Creates a Lasso regressor with regularization strength `alpha`.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha` is negative, NaN, or infinite.
    pub fn new(alpha: f64) -> Result<Self> {
        validate_alpha(alpha)?;
        Ok(Self {
            alpha,
            fit_intercept: true,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
        })
    }

    /// Enables or disables intercept fitting.
    #[must_use]
    pub const fn with_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }

    /// Sets the maximum number of coordinate-descent passes over every
    /// feature.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_iterations` is zero.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Result<Self> {
        validate_max_iterations(max_iterations)?;
        self.max_iterations = max_iterations;
        Ok(self)
    }

    /// Sets the convergence tolerance applied to the largest coefficient
    /// change in a full coordinate-descent pass.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is non-positive, NaN, or infinite.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self> {
        validate_tolerance(tolerance)?;
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Returns the L1 regularization strength.
    #[must_use]
    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    /// Returns whether an unpenalized intercept will be fitted.
    #[must_use]
    pub const fn fit_intercept(self) -> bool {
        self.fit_intercept
    }

    /// Returns the configured iteration budget.
    #[must_use]
    pub const fn max_iterations(self) -> usize {
        self.max_iterations
    }

    /// Returns the configured convergence tolerance.
    #[must_use]
    pub const fn tolerance(self) -> f64 {
        self.tolerance
    }

    /// Fits Lasso regression by coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid regularization, an invalid iteration
    /// budget or tolerance, when features are empty or non-finite, when a
    /// target is non-finite, or when the solver fails to converge within
    /// the configured iteration budget.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedLassoRegression> {
        validate_alpha(self.alpha)?;
        validate_max_iterations(self.max_iterations)?;
        validate_tolerance(self.tolerance)?;
        let (coefficients, intercept, convergence) = fit_coordinate_descent(
            dataset,
            self.fit_intercept,
            self.alpha,
            1.0,
            self.max_iterations,
            self.tolerance,
        )?;
        Ok(FittedLassoRegression {
            coefficients,
            intercept,
            alpha: self.alpha,
            convergence,
        })
    }
}

impl Fit<&Dataset<f64>, ()> for LassoRegression {
    type Fitted = FittedLassoRegression;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Coefficients learned by [`LassoRegression`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedLassoRegression {
    coefficients: Array1<f64>,
    intercept: f64,
    alpha: f64,
    convergence: ConvergenceReport,
}

impl FittedLassoRegression {
    /// Returns one coefficient per input feature.
    #[must_use]
    pub const fn coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }

    /// Returns the fitted intercept, or zero when intercept fitting was
    /// disabled.
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

    /// Returns how the coordinate-descent solver converged.
    #[must_use]
    pub const fn convergence(&self) -> &ConvergenceReport {
        &self.convergence
    }

    /// Returns the number of features with a nonzero fitted coefficient.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn n_nonzero_coefficients(&self) -> usize {
        self.coefficients
            .iter()
            .filter(|&&value| value != 0.0)
            .count()
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

impl<'a> Predict<ArrayView2<'a, f64>> for FittedLassoRegression {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}
