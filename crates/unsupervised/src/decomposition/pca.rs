// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView2, Axis};

use machlearn_core::core::{MlError, Result, Transform, validate_feature_count, validate_features};
use machlearn_preprocessing::preprocessing::{FittedTransformer, TransformerEstimator};

/// Configures principal component analysis.
///
/// Components are the eigenvectors of the training data's sample covariance
/// matrix, ordered by decreasing explained variance. Each component's sign is
/// fixed so its largest-magnitude entry is positive, making output
/// deterministic regardless of the underlying solver's arbitrary sign choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrincipalComponentAnalysis {
    n_components: Option<usize>,
}

impl PrincipalComponentAnalysis {
    /// Creates a PCA estimator that keeps every available component.
    #[must_use]
    pub const fn new() -> Self {
        Self { n_components: None }
    }

    /// Limits how many principal components are retained.
    ///
    /// `None` keeps every component the training data supports
    /// (`min(n_samples - 1, n_features)`).
    ///
    /// # Errors
    ///
    /// Returns an error when `n_components` is `Some(0)`.
    pub fn with_n_components(mut self, n_components: Option<usize>) -> Result<Self> {
        if let Some(0) = n_components {
            return Err(MlError::InvalidComponentCount(0));
        }
        self.n_components = n_components;
        Ok(self)
    }

    /// Returns the configured component limit.
    #[must_use]
    pub const fn n_components(self) -> Option<usize> {
        self.n_components
    }

    /// Learns principal components from the sample covariance of `records`.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty or non-finite, when there
    /// are fewer than two samples, or when more components are requested
    /// than the training data supports.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedPrincipalComponentAnalysis> {
        validate_features(records)?;
        let n_samples = records.nrows();
        let n_features = records.ncols();
        if n_samples < 2 {
            return Err(MlError::InsufficientSamples {
                required: 2,
                actual: n_samples,
            });
        }

        let maximum_components = n_features.min(n_samples - 1);
        let requested_components = self.n_components.unwrap_or(maximum_components);
        if requested_components > maximum_components {
            return Err(MlError::TooManyComponents {
                requested: requested_components,
                maximum: maximum_components,
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let sample_count = n_samples as f64;
        let mean = Array1::from_iter(
            records
                .axis_iter(Axis(1))
                .map(|column| column.sum() / sample_count),
        );
        let mut centered = records.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        #[allow(clippy::cast_precision_loss)]
        let degrees_of_freedom = (n_samples - 1) as f64;
        let covariance = Mat::from_fn(n_features, n_features, |row, column| {
            centered.column(row).dot(&centered.column(column)) / degrees_of_freedom
        });
        let eigen = covariance
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_error| MlError::NonFiniteSolverOutput { index: 0 })?;

        // faer returns eigenpairs sorted in ascending order; PCA wants
        // decreasing explained variance, so pairs are read back to front.
        let mut explained_variance = Array1::zeros(requested_components);
        let mut components = Array2::zeros((requested_components, n_features));
        for component_index in 0..requested_components {
            let source_column = n_features - 1 - component_index;
            let eigenvalue = eigen.S().column_vector()[source_column].max(0.0);
            explained_variance[component_index] = eigenvalue;

            let sign = dominant_sign(&eigen, source_column, n_features);
            for feature_index in 0..n_features {
                components[[component_index, feature_index]] =
                    sign * eigen.U()[(feature_index, source_column)];
            }
        }

        let total_variance: f64 = (0..n_features)
            .map(|index| eigen.S().column_vector()[index].max(0.0))
            .sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|value| value / total_variance)
        } else {
            Array1::zeros(requested_components)
        };

        Ok(FittedPrincipalComponentAnalysis {
            components,
            mean,
            explained_variance,
            explained_variance_ratio,
        })
    }
}

/// Returns `1.0` if the eigenvector's largest-magnitude entry is already
/// positive, or `-1.0` to flip it, fixing an otherwise arbitrary sign.
fn dominant_sign(
    eigen: &faer::linalg::solvers::SelfAdjointEigen<f64>,
    column: usize,
    n_features: usize,
) -> f64 {
    let mut dominant_value = 0.0_f64;
    let mut dominant_magnitude = 0.0_f64;
    for feature_index in 0..n_features {
        let value = eigen.U()[(feature_index, column)];
        if value.abs() > dominant_magnitude {
            dominant_magnitude = value.abs();
            dominant_value = value;
        }
    }
    if dominant_value < 0.0 { -1.0 } else { 1.0 }
}

/// Principal components learned by [`PrincipalComponentAnalysis`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedPrincipalComponentAnalysis {
    components: Array2<f64>,
    mean: Array1<f64>,
    explained_variance: Array1<f64>,
    explained_variance_ratio: Array1<f64>,
}

impl FittedPrincipalComponentAnalysis {
    /// Returns one unit-norm principal axis per row, in decreasing
    /// explained-variance order.
    #[must_use]
    pub const fn components(&self) -> &Array2<f64> {
        &self.components
    }

    /// Returns the per-feature offset subtracted before projecting.
    #[must_use]
    pub const fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// Returns the training-data variance captured along each component.
    #[must_use]
    pub const fn explained_variance(&self) -> &Array1<f64> {
        &self.explained_variance
    }

    /// Returns the fraction of total training-data variance captured by
    /// each component.
    #[must_use]
    pub const fn explained_variance_ratio(&self) -> &Array1<f64> {
        &self.explained_variance_ratio
    }

    /// Returns the number of retained components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.components.nrows()
    }

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.components.ncols()
    }

    /// Projects centered records onto the fitted principal components.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;
        let mut centered = records.to_owned();
        for mut row in centered.rows_mut() {
            row -= &self.mean;
        }
        Ok(centered.dot(&self.components.t()))
    }

    /// Reconstructs records in the original feature space from their
    /// projected representation.
    ///
    /// The reconstruction is exact only when every available component was
    /// retained; otherwise it is the best approximation the retained
    /// components support.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is empty, non-finite, or does not
    /// have one column per retained component.
    pub fn inverse_transform(&self, projected: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(projected)?;
        validate_feature_count(projected.ncols(), self.n_components())?;
        let mut reconstructed = projected.dot(&self.components);
        for mut row in reconstructed.rows_mut() {
            row += &self.mean;
        }
        Ok(reconstructed)
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedPrincipalComponentAnalysis {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

impl TransformerEstimator for PrincipalComponentAnalysis {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedPrincipalComponentAnalysis {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}
