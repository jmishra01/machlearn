// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView2};

use crate::neighbors::common::{
    Weighting, nearest_neighbors, neighbor_weights, validate_n_neighbors, validate_training_size,
};
use machlearn_core::core::{
    Dataset, Fit, MlError, Predict, Result, validate_feature_count, validate_features,
};

/// Configures a k-nearest-neighbors regressor.
///
/// Prediction averages the targets of the `n_neighbors` training points
/// closest to a query point under Euclidean distance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KNeighborsRegressor {
    n_neighbors: usize,
    weighting: Weighting,
}

impl Default for KNeighborsRegressor {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weighting: Weighting::Uniform,
        }
    }
}

impl KNeighborsRegressor {
    /// Creates a regressor that averages the `n_neighbors` closest training
    /// targets.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_neighbors` is zero.
    pub fn new(n_neighbors: usize) -> Result<Self> {
        validate_n_neighbors(n_neighbors)?;
        Ok(Self {
            n_neighbors,
            weighting: Weighting::Uniform,
        })
    }

    /// Sets how neighbor targets are weighted by distance.
    #[must_use]
    pub const fn with_weighting(mut self, weighting: Weighting) -> Self {
        self.weighting = weighting;
        self
    }

    /// Returns the configured neighbor count.
    #[must_use]
    pub const fn n_neighbors(self) -> usize {
        self.n_neighbors
    }

    /// Returns the configured target weighting.
    #[must_use]
    pub const fn weighting(self) -> Weighting {
        self.weighting
    }

    /// Stores the training data used for later neighbor lookups.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_neighbors` is zero, when the training set has
    /// fewer samples than `n_neighbors`, when features are empty or
    /// non-finite, or when a target is non-finite.
    pub fn fit(&self, dataset: &Dataset<f64>) -> Result<FittedKNeighborsRegressor> {
        validate_n_neighbors(self.n_neighbors)?;
        validate_training_size(self.n_neighbors, dataset.n_samples())?;
        validate_features(dataset.records())?;
        for (index, &target) in dataset.targets().iter().enumerate() {
            if !target.is_finite() {
                return Err(MlError::NonFiniteActualTarget { index });
            }
        }
        Ok(FittedKNeighborsRegressor {
            records: dataset.records().to_owned(),
            targets: dataset.targets().to_owned(),
            n_neighbors: self.n_neighbors,
            weighting: self.weighting,
        })
    }
}

impl Fit<&Dataset<f64>, ()> for KNeighborsRegressor {
    type Fitted = FittedKNeighborsRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Training data retained by [`KNeighborsRegressor`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedKNeighborsRegressor {
    records: Array2<f64>,
    targets: Array1<f64>,
    n_neighbors: usize,
    weighting: Weighting,
}

impl FittedKNeighborsRegressor {
    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.records.ncols()
    }

    /// Returns the number of stored training samples.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.records.nrows()
    }

    /// Returns the configured neighbor count.
    #[must_use]
    pub const fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Returns the configured target weighting.
    #[must_use]
    pub const fn weighting(&self) -> Weighting {
        self.weighting
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

        let mut predictions = Array1::zeros(records.nrows());
        for (row_index, row) in records.rows().into_iter().enumerate() {
            let neighbors = nearest_neighbors(self.records.view(), row, self.n_neighbors);
            let weights = neighbor_weights(self.weighting, &neighbors);
            let weight_sum: f64 = weights.iter().sum();
            let weighted_target: f64 = neighbors
                .iter()
                .zip(weights.iter())
                .map(|(&(neighbor_index, _distance), &weight)| {
                    weight * self.targets[neighbor_index]
                })
                .sum();
            let prediction = weighted_target / weight_sum;
            if !prediction.is_finite() {
                return Err(MlError::NonFinitePrediction { index: row_index });
            }
            predictions[row_index] = prediction;
        }
        Ok(predictions)
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedKNeighborsRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}
