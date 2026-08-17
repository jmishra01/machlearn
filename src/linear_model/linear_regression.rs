// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{
    core::{Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features},
    solver::solve_least_squares,
};

/// Configures ordinary least-squares linear regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearRegression {
    fit_intercept: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
        }
    }
}

impl LinearRegression {
    /// Creates a linear regressor that fits an intercept.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fit_intercept: true,
        }
    }

    /// Enables or disables intercept fitting.
    #[must_use]
    pub const fn with_intercept(mut self, enabled: bool) -> Self {
        self.fit_intercept = enabled;
        self
    }

    /// Returns whether an intercept will be fitted.
    #[must_use]
    pub const fn fit_intercept(self) -> bool {
        self.fit_intercept
    }

    /// Fits an ordinary least-squares model using column-pivoted QR.
    ///
    /// # Errors
    ///
    /// Returns an error when targets are non-finite, there are fewer samples
    /// than fitted parameters, or the augmented design matrix is numerically
    /// rank deficient.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedLinearRegression> {
        if self.fit_intercept {
            let mut design = Array2::ones((dataset.n_samples(), dataset.n_features() + 1));
            for (feature_index, feature) in dataset.records().axis_iter(Axis(1)).enumerate() {
                design.column_mut(feature_index + 1).assign(&feature);
            }
            let parameters = solve_least_squares(design.view(), dataset.targets())?;
            Ok(FittedLinearRegression {
                intercept: parameters[0],
                coefficients: Array1::from_iter(parameters.iter().skip(1).copied()),
            })
        } else {
            let coefficients = solve_least_squares(dataset.records(), dataset.targets())?;
            Ok(FittedLinearRegression {
                coefficients,
                intercept: 0.0,
            })
        }
    }
}

impl Fit<&Dataset<f64>, ()> for LinearRegression {
    type Fitted = FittedLinearRegression;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Coefficients learned by [`LinearRegression`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedLinearRegression {
    coefficients: Array1<f64>,
    intercept: f64,
}

impl FittedLinearRegression {
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
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;
        let mut predictions = records.dot(&self.coefficients);
        predictions += self.intercept;
        if let Some((index, _prediction)) = predictions
            .iter()
            .enumerate()
            .find(|(_index, prediction)| !prediction.is_finite())
        {
            return Err(MlError::NonFinitePrediction { index });
        }
        Ok(predictions)
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedLinearRegression {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}
