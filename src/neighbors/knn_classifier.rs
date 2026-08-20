// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use std::collections::BTreeMap;

use ndarray::{Array1, Array2, ArrayView2};

use crate::{
    core::{Dataset, Fit, Predict, Result, validate_feature_count, validate_features},
    neighbors::common::{
        Weighting, nearest_neighbors, neighbor_weights, validate_n_neighbors,
        validate_training_size,
    },
};

/// Configures a k-nearest-neighbors classifier.
///
/// Prediction votes among the `n_neighbors` training points closest to a
/// query point under Euclidean distance. Classes are stored in sorted order,
/// making predictions deterministic even when training rows are reordered.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KNeighborsClassifier {
    n_neighbors: usize,
    weighting: Weighting,
}

impl Default for KNeighborsClassifier {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weighting: Weighting::Uniform,
        }
    }
}

impl KNeighborsClassifier {
    /// Creates a classifier that votes among the `n_neighbors` closest
    /// training points.
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

    /// Sets how neighbor votes are weighted by distance.
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

    /// Returns the configured vote weighting.
    #[must_use]
    pub const fn weighting(self) -> Weighting {
        self.weighting
    }

    /// Stores the training data used for later neighbor lookups.
    ///
    /// Labels may be any cloneable ordered type.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_neighbors` is zero, when the training set has
    /// fewer samples than `n_neighbors`, or when features are empty or
    /// non-finite.
    pub fn fit<Label>(&self, dataset: &Dataset<Label>) -> Result<FittedKNeighborsClassifier<Label>>
    where
        Label: Clone + Ord,
    {
        validate_n_neighbors(self.n_neighbors)?;
        validate_training_size(self.n_neighbors, dataset.n_samples())?;
        validate_features(dataset.records())?;
        Ok(FittedKNeighborsClassifier {
            records: dataset.records().to_owned(),
            labels: dataset.targets().to_vec(),
            n_neighbors: self.n_neighbors,
            weighting: self.weighting,
        })
    }
}

impl<Label> Fit<&Dataset<Label>, ()> for KNeighborsClassifier
where
    Label: Clone + Ord,
{
    type Fitted = FittedKNeighborsClassifier<Label>;

    fn fit(&self, dataset: &Dataset<Label>, (): ()) -> Result<Self::Fitted> {
        Self::fit(self, dataset)
    }
}

/// Training data retained by [`KNeighborsClassifier`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedKNeighborsClassifier<Label> {
    records: Array2<f64>,
    labels: Vec<Label>,
    n_neighbors: usize,
    weighting: Weighting,
}

impl<Label> FittedKNeighborsClassifier<Label> {
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

    /// Returns the configured vote weighting.
    #[must_use]
    pub const fn weighting(&self) -> Weighting {
        self.weighting
    }
}

impl<Label> FittedKNeighborsClassifier<Label>
where
    Label: Clone + Ord,
{
    /// Predicts class labels for a feature matrix.
    ///
    /// Vote ties are resolved in favor of the smallest sorted label.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<Label>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;

        let mut predictions = Vec::with_capacity(records.nrows());
        for row in records.rows() {
            let neighbors = nearest_neighbors(self.records.view(), row, self.n_neighbors);
            let weights = neighbor_weights(self.weighting, &neighbors);

            let mut votes: BTreeMap<Label, f64> = BTreeMap::new();
            for (&(neighbor_index, _distance), &weight) in neighbors.iter().zip(weights.iter()) {
                *votes
                    .entry(self.labels[neighbor_index].clone())
                    .or_insert(0.0) += weight;
            }

            // `votes` always holds at least one entry: fitting requires the
            // training set to have at least `n_neighbors >= 1` samples, so
            // every query produces at least one neighbor vote. Reducing in
            // `BTreeMap` order with a strict `>` comparison favors the
            // smallest label on ties.
            if let Some((label, _vote)) = votes.into_iter().reduce(|current_best, candidate| {
                if candidate.1 > current_best.1 {
                    candidate
                } else {
                    current_best
                }
            }) {
                predictions.push(label);
            }
        }
        Ok(Array1::from_vec(predictions))
    }
}

impl<'a, Label> Predict<ArrayView2<'a, f64>> for FittedKNeighborsClassifier<Label>
where
    Label: Clone + Ord,
{
    type Output = Array1<Label>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}
