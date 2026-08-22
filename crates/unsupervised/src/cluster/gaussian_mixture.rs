// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use std::f64::consts::PI;

use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::cluster::kmeans::KMeans;
use machlearn_core::core::{MlError, Predict, Result, validate_feature_count, validate_features};

const DEFAULT_MAX_ITERATIONS: usize = 100;
const DEFAULT_TOLERANCE: f64 = 1.0e-3;
const DEFAULT_REG_COVAR: f64 = 1.0e-6;
const DEFAULT_SEED: u64 = 42;
const MINIMUM_EIGENVALUE: f64 = 1.0e-12;
const MINIMUM_EFFECTIVE_COUNT: f64 = 1.0e-12;

/// Configures a Gaussian mixture model fit by expectation-maximization.
///
/// Every component is a full-covariance multivariate Gaussian. Initial
/// component weights, means, and covariances come from a hard
/// [`crate::cluster::KMeans`] clustering of the training data; expectation-maximization
/// then alternates computing each row's per-component responsibility (the
/// posterior probability it belongs to that component) and re-estimating
/// every component's weight, mean, and covariance from those
/// responsibilities, stopping once the average per-sample log-likelihood
/// improves by less than `tolerance` or `max_iterations` is reached.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianMixture {
    n_components: usize,
    max_iterations: usize,
    tolerance: f64,
    reg_covar: f64,
    seed: u64,
}

impl GaussianMixture {
    /// Creates a Gaussian mixture estimator with `n_components` full-covariance
    /// components.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_components` is zero.
    pub fn new(n_components: usize) -> Result<Self> {
        validate_n_components(n_components)?;
        Ok(Self {
            n_components,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            reg_covar: DEFAULT_REG_COVAR,
            seed: DEFAULT_SEED,
        })
    }

    /// Sets the maximum number of expectation-maximization iterations.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_iterations` is zero.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Result<Self> {
        validate_max_iterations(max_iterations)?;
        self.max_iterations = max_iterations;
        Ok(self)
    }

    /// Sets the average log-likelihood improvement that stops iteration
    /// early.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is non-positive, NaN, or infinite.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self> {
        validate_tolerance(tolerance)?;
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Sets the value added to every component covariance's diagonal,
    /// preventing singular covariances when a component collapses onto too
    /// few points.
    ///
    /// # Errors
    ///
    /// Returns an error when `reg_covar` is negative, NaN, or infinite.
    pub fn with_reg_covar(mut self, reg_covar: f64) -> Result<Self> {
        validate_reg_covar(reg_covar)?;
        self.reg_covar = reg_covar;
        Ok(self)
    }

    /// Sets the deterministic seed used for the k-means initialization.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the configured component count.
    #[must_use]
    pub const fn n_components(self) -> usize {
        self.n_components
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

    /// Returns the configured covariance regularization.
    #[must_use]
    pub const fn reg_covar(self) -> f64 {
        self.reg_covar
    }

    /// Returns the configured random seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Fits component weights, means, and covariances by
    /// expectation-maximization.
    ///
    /// Failing to converge within `max_iterations` is not an error.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid component count, iteration budget,
    /// tolerance, or covariance regularization, when features are empty or
    /// non-finite, or when there are fewer samples than components.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedGaussianMixture> {
        validate_n_components(self.n_components)?;
        validate_max_iterations(self.max_iterations)?;
        validate_tolerance(self.tolerance)?;
        validate_reg_covar(self.reg_covar)?;
        validate_features(records)?;
        if records.nrows() < self.n_components {
            return Err(MlError::InsufficientSamples {
                required: self.n_components,
                actual: records.nrows(),
            });
        }

        let n_samples = records.nrows();

        let init_assignments = KMeans::new(self.n_components)?
            .with_seed(self.seed)
            .fit(records)?
            .predict(records)?;
        let (mut weights, mut means, mut covariances) = initialize_components(
            records,
            self.n_components,
            &init_assignments,
            self.reg_covar,
        );

        let mut responsibilities = Array2::zeros((n_samples, self.n_components));
        let mut average_log_likelihood = f64::NEG_INFINITY;
        let mut converged = false;
        let mut iterations = 0;

        for _iteration in 0..self.max_iterations {
            iterations += 1;

            let total_log_likelihood = e_step(
                records,
                &weights,
                &means,
                &covariances,
                &mut responsibilities,
            )?;
            #[allow(clippy::cast_precision_loss)]
            let updated_average_log_likelihood = total_log_likelihood / n_samples as f64;

            (weights, means, covariances) = m_step(
                records,
                &responsibilities,
                self.n_components,
                self.reg_covar,
            );

            let change = updated_average_log_likelihood - average_log_likelihood;
            average_log_likelihood = updated_average_log_likelihood;
            if change.abs() < self.tolerance {
                converged = true;
                break;
            }
        }

        Ok(FittedGaussianMixture {
            weights,
            means,
            covariances,
            n_features: records.ncols(),
            converged,
            n_iterations: iterations,
            log_likelihood: average_log_likelihood,
        })
    }
}

/// Builds each component's initial weight, mean, and covariance from a hard
/// k-means assignment.
///
/// A component that k-means left empty falls back to the overall dataset
/// mean, covariance, and a uniform weight, so expectation-maximization
/// still has a well-defined starting point for it.
fn initialize_components(
    records: ArrayView2<'_, f64>,
    n_components: usize,
    assignments: &Array1<usize>,
    reg_covar: f64,
) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let n_features = records.ncols();
    #[allow(clippy::cast_precision_loss)]
    let sample_count = records.nrows() as f64;
    let overall_mean = column_mean(records);
    let overall_covariance = empirical_covariance(records, overall_mean.view(), reg_covar);

    let mut weights = Array1::zeros(n_components);
    let mut means = Array2::zeros((n_components, n_features));
    let mut covariances: Vec<Array2<f64>> = Vec::with_capacity(n_components);
    for component in 0..n_components {
        let rows: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(row, &cluster)| (cluster == component).then_some(row))
            .collect();
        if rows.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            let uniform_weight = 1.0 / n_components as f64;
            weights[component] = uniform_weight;
            means.row_mut(component).assign(&overall_mean);
            covariances.push(overall_covariance.clone());
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let weight = rows.len() as f64 / sample_count;
        weights[component] = weight;
        let component_records = records.select(Axis(0), &rows);
        let mean = column_mean(component_records.view());
        covariances.push(empirical_covariance(
            component_records.view(),
            mean.view(),
            reg_covar,
        ));
        means.row_mut(component).assign(&mean);
    }
    (weights, means, covariances)
}

/// Computes every row's per-component responsibility under the current
/// parameters, writing it into `responsibilities`, and returns the total
/// (summed, not averaged) log-likelihood of `records` under those
/// parameters.
fn e_step(
    records: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    means: &Array2<f64>,
    covariances: &[Array2<f64>],
    responsibilities: &mut Array2<f64>,
) -> Result<f64> {
    let n_components = weights.len();
    let factors = factorize_components(covariances)?;

    let mut total_log_likelihood = 0.0;
    for (row_index, row) in records.rows().into_iter().enumerate() {
        let log_probabilities: Array1<f64> =
            Array1::from_iter((0..n_components).map(|component| {
                let (eigenvectors, eigenvalues) = &factors[component];
                weights[component].ln()
                    + gaussian_log_density(
                        row,
                        means.row(component),
                        eigenvectors.view(),
                        eigenvalues.view(),
                    )
            }));
        let log_sum = log_sum_exp(log_probabilities.view());
        total_log_likelihood += log_sum;
        for component in 0..n_components {
            responsibilities[[row_index, component]] =
                (log_probabilities[component] - log_sum).exp();
        }
    }
    Ok(total_log_likelihood)
}

/// Re-estimates every component's weight, mean, and covariance from the
/// current responsibilities.
fn m_step(
    records: ArrayView2<'_, f64>,
    responsibilities: &Array2<f64>,
    n_components: usize,
    reg_covar: f64,
) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let n_features = records.ncols();
    #[allow(clippy::cast_precision_loss)]
    let sample_count = records.nrows() as f64;

    let mut weights = Array1::zeros(n_components);
    let mut means = Array2::zeros((n_components, n_features));
    let mut covariances: Vec<Array2<f64>> = Vec::with_capacity(n_components);
    for component in 0..n_components {
        let responsibility = responsibilities.column(component);
        let effective_count = responsibility.sum().max(MINIMUM_EFFECTIVE_COUNT);
        weights[component] = effective_count / sample_count;

        let mut mean = Array1::zeros(n_features);
        for (row_index, row) in records.rows().into_iter().enumerate() {
            mean.scaled_add(responsibility[row_index], &row);
        }
        mean /= effective_count;

        let mut covariance = Array2::zeros((n_features, n_features));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            let point_weight = responsibility[row_index];
            let centered = &row.to_owned() - &mean;
            for feature_row in 0..n_features {
                for feature_column in 0..n_features {
                    covariance[[feature_row, feature_column]] +=
                        point_weight * centered[feature_row] * centered[feature_column];
                }
            }
        }
        covariance /= effective_count;
        for feature_index in 0..n_features {
            covariance[[feature_index, feature_index]] += reg_covar;
        }

        means.row_mut(component).assign(&mean);
        covariances.push(covariance);
    }
    (weights, means, covariances)
}

/// Component weights, means, and covariances learned by [`GaussianMixture`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedGaussianMixture {
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    n_features: usize,
    converged: bool,
    n_iterations: usize,
    log_likelihood: f64,
}

impl FittedGaussianMixture {
    /// Returns the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.weights.len()
    }

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns each component's mixing weight, in component order.
    #[must_use]
    pub const fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    /// Returns one mean row per component.
    #[must_use]
    pub const fn means(&self) -> &Array2<f64> {
        &self.means
    }

    /// Returns one full covariance matrix per component, in component
    /// order.
    #[must_use]
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.covariances
    }

    /// Returns whether the average log-likelihood improvement fell below
    /// the configured tolerance before the iteration budget was exhausted.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the number of expectation-maximization iterations completed.
    #[must_use]
    pub const fn n_iterations(&self) -> usize {
        self.n_iterations
    }

    /// Returns the training data's average per-sample log-likelihood under
    /// the fitted mixture.
    #[must_use]
    pub const fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Computes each row's per-sample log-likelihood under the fitted
    /// mixture, marginalizing over every component.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or a component's covariance cannot be factorized.
    pub fn score_samples(&self, records: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let factors = factorize_components(&self.covariances)?;
        let mut scores = Array1::zeros(records.nrows());
        for (row_index, row) in records.rows().into_iter().enumerate() {
            let log_probabilities: Array1<f64> =
                Array1::from_iter((0..self.n_components()).map(|component| {
                    let (eigenvectors, eigenvalues) = &factors[component];
                    self.weights[component].ln()
                        + gaussian_log_density(
                            row,
                            self.means.row(component),
                            eigenvectors.view(),
                            eigenvalues.view(),
                        )
                }));
            scores[row_index] = log_sum_exp(log_probabilities.view());
        }
        Ok(scores)
    }

    /// Predicts normalized component-membership probabilities for every
    /// row.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::score_samples`].
    pub fn predict_probabilities(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let factors = factorize_components(&self.covariances)?;
        let mut probabilities = Array2::zeros((records.nrows(), self.n_components()));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            let log_probabilities: Array1<f64> =
                Array1::from_iter((0..self.n_components()).map(|component| {
                    let (eigenvectors, eigenvalues) = &factors[component];
                    self.weights[component].ln()
                        + gaussian_log_density(
                            row,
                            self.means.row(component),
                            eigenvectors.view(),
                            eigenvalues.view(),
                        )
                }));
            let log_sum = log_sum_exp(log_probabilities.view());
            for component in 0..self.n_components() {
                probabilities[[row_index, component]] =
                    (log_probabilities[component] - log_sum).exp();
            }
        }
        Ok(probabilities)
    }

    /// Predicts the component with the greatest membership probability for
    /// every row.
    ///
    /// Ties are resolved in favor of the lowest component index.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::score_samples`].
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        let probabilities = self.predict_probabilities(records)?;
        Ok(Array1::from_iter(probabilities.rows().into_iter().map(
            |row| {
                let mut best_component = 0;
                for component in 1..row.len() {
                    if row[component] > row[best_component] {
                        best_component = component;
                    }
                }
                best_component
            },
        )))
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedGaussianMixture {
    type Output = Array1<usize>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

fn column_mean(records: ArrayView2<'_, f64>) -> Array1<f64> {
    #[allow(clippy::cast_precision_loss)]
    let count = records.nrows() as f64;
    records.sum_axis(Axis(0)) / count
}

fn empirical_covariance(
    records: ArrayView2<'_, f64>,
    mean: ArrayView1<'_, f64>,
    reg_covar: f64,
) -> Array2<f64> {
    let n_features = records.ncols();
    #[allow(clippy::cast_precision_loss)]
    let count = records.nrows() as f64;
    let mut covariance = Array2::zeros((n_features, n_features));
    for row in records.rows() {
        let centered = &row.to_owned() - &mean;
        for feature_row in 0..n_features {
            for feature_column in 0..n_features {
                covariance[[feature_row, feature_column]] +=
                    centered[feature_row] * centered[feature_column];
            }
        }
    }
    covariance /= count;
    for feature_index in 0..n_features {
        covariance[[feature_index, feature_index]] += reg_covar;
    }
    covariance
}

/// Eigendecomposes every component's covariance matrix, returning
/// `(eigenvectors, eigenvalues)` pairs in component order.
///
/// The eigenbasis lets the Gaussian log-density be computed as a sum over
/// independent directions without ever forming an explicit matrix inverse:
/// projecting a centered point onto the eigenvectors diagonalizes the
/// quadratic form, so each direction's contribution is just the squared
/// projection divided by its eigenvalue.
fn factorize_components(covariances: &[Array2<f64>]) -> Result<Vec<(Array2<f64>, Array1<f64>)>> {
    covariances
        .iter()
        .map(|covariance| factorize(covariance.view()))
        .collect()
}

fn factorize(covariance: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array1<f64>)> {
    let n_features = covariance.nrows();
    let matrix = Mat::from_fn(n_features, n_features, |row, column| {
        covariance[[row, column]]
    });
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_error| MlError::NonFiniteSolverOutput { index: 0 })?;

    let eigenvalues = Array1::from_iter(
        (0..n_features).map(|index| eigen.S().column_vector()[index].max(MINIMUM_EIGENVALUE)),
    );
    let eigenvectors = Array2::from_shape_fn((n_features, n_features), |(row, column)| {
        eigen.U()[(row, column)]
    });
    Ok((eigenvectors, eigenvalues))
}

/// Log-density of a multivariate Gaussian with the covariance already
/// factorized into `(eigenvectors, eigenvalues)` by [`factorize`].
fn gaussian_log_density(
    point: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    eigenvectors: ArrayView2<'_, f64>,
    eigenvalues: ArrayView1<'_, f64>,
) -> f64 {
    let n_features = point.len();
    let centered = &point.to_owned() - &mean;
    let mut log_determinant = 0.0;
    let mut mahalanobis = 0.0;
    for component_index in 0..n_features {
        let eigenvalue = eigenvalues[component_index];
        log_determinant += eigenvalue.ln();
        let projection: f64 = (0..n_features)
            .map(|feature_index| {
                eigenvectors[[feature_index, component_index]] * centered[feature_index]
            })
            .sum();
        mahalanobis += projection * projection / eigenvalue;
    }
    #[allow(clippy::cast_precision_loss)]
    let dimension = n_features as f64;
    -0.5 * (dimension * (2.0 * PI).ln() + log_determinant + mahalanobis)
}

fn log_sum_exp(values: ArrayView1<'_, f64>) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|&value| (value - maximum).exp()).sum();
    maximum + sum.ln()
}

fn validate_n_components(n_components: usize) -> Result<()> {
    if n_components == 0 {
        return Err(MlError::InvalidComponentCount(n_components));
    }
    Ok(())
}

fn validate_max_iterations(max_iterations: usize) -> Result<()> {
    if max_iterations == 0 {
        return Err(MlError::InvalidMaxIterations(max_iterations));
    }
    Ok(())
}

fn validate_tolerance(tolerance: f64) -> Result<()> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MlError::InvalidTolerance(tolerance));
    }
    Ok(())
}

fn validate_reg_covar(reg_covar: f64) -> Result<()> {
    if !reg_covar.is_finite() || reg_covar < 0.0 {
        return Err(MlError::InvalidRegularization(reg_covar));
    }
    Ok(())
}
