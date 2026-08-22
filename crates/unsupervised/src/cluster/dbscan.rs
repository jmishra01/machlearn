// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use std::collections::{HashSet, VecDeque};

use ndarray::{Array1, ArrayView1, ArrayView2};

use machlearn_core::core::{MlError, Result, validate_features};

/// Configures density-based spatial clustering (DBSCAN).
///
/// A point is a *core point* when at least `min_samples` points (counting
/// itself) lie within Euclidean distance `eps` of it. Clusters grow by
/// chaining core points that lie within `eps` of one another; any point
/// reachable from a core point (directly within `eps`, or transitively
/// through a chain of core points) joins that core point's cluster as a
/// *border point*. Points reachable from no core point are labeled noise.
/// Unlike [`crate::cluster::KMeans`], the number of clusters is discovered rather
/// than configured, and clusters may take any shape, not just convex
/// regions around a centroid.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DBSCAN {
    eps: f64,
    min_samples: usize,
}

impl DBSCAN {
    /// Creates a DBSCAN estimator with neighborhood radius `eps` and a
    /// minimum of `min_samples` points (counting the point itself) required
    /// for a point to seed or extend a cluster.
    ///
    /// # Errors
    ///
    /// Returns an error when `eps` is non-positive, NaN, or infinite, or
    /// when `min_samples` is zero.
    pub fn new(eps: f64, min_samples: usize) -> Result<Self> {
        validate_eps(eps)?;
        validate_min_samples(min_samples)?;
        Ok(Self { eps, min_samples })
    }

    /// Sets the neighborhood radius.
    ///
    /// # Errors
    ///
    /// Returns an error when `eps` is non-positive, NaN, or infinite.
    pub fn with_eps(mut self, eps: f64) -> Result<Self> {
        validate_eps(eps)?;
        self.eps = eps;
        Ok(self)
    }

    /// Sets the minimum neighborhood size (counting the point itself)
    /// required for a point to seed or extend a cluster.
    ///
    /// # Errors
    ///
    /// Returns an error when `min_samples` is zero.
    pub fn with_min_samples(mut self, min_samples: usize) -> Result<Self> {
        validate_min_samples(min_samples)?;
        self.min_samples = min_samples;
        Ok(self)
    }

    /// Returns the configured neighborhood radius.
    #[must_use]
    pub const fn eps(self) -> f64 {
        self.eps
    }

    /// Returns the configured minimum neighborhood size.
    #[must_use]
    pub const fn min_samples(self) -> usize {
        self.min_samples
    }

    /// Clusters `records` by chaining density-reachable core points.
    ///
    /// DBSCAN is transductive: it only labels the rows it was fit on, so
    /// the fitted model has no `predict` for unseen rows.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid `eps` or `min_samples`, or when
    /// features are empty or non-finite.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedDBSCAN> {
        validate_eps(self.eps)?;
        validate_min_samples(self.min_samples)?;
        validate_features(records)?;

        let n_samples = records.nrows();
        let radius_squared = self.eps * self.eps;
        let neighbors: Vec<Vec<usize>> = records
            .rows()
            .into_iter()
            .map(|row| {
                records
                    .rows()
                    .into_iter()
                    .enumerate()
                    .filter(|(_, other)| squared_distance(row, *other) <= radius_squared)
                    .map(|(index, _)| index)
                    .collect()
            })
            .collect();

        let mut labels: Vec<Option<usize>> = vec![None; n_samples];
        let mut visited = vec![false; n_samples];
        let mut n_clusters = 0;

        for seed in 0..n_samples {
            if visited[seed] || neighbors[seed].len() < self.min_samples {
                continue;
            }

            let cluster = n_clusters;
            n_clusters += 1;
            visited[seed] = true;
            labels[seed] = Some(cluster);

            let mut queue: VecDeque<usize> = neighbors[seed].iter().copied().collect();
            let mut queued: HashSet<usize> = queue.iter().copied().collect();
            while let Some(point) = queue.pop_front() {
                if !visited[point] {
                    visited[point] = true;
                    if neighbors[point].len() >= self.min_samples {
                        for &candidate in &neighbors[point] {
                            if queued.insert(candidate) {
                                queue.push_back(candidate);
                            }
                        }
                    }
                }
                if labels[point].is_none() {
                    labels[point] = Some(cluster);
                }
            }
        }

        Ok(FittedDBSCAN {
            labels: Array1::from_vec(labels),
            n_clusters,
            n_features: records.ncols(),
        })
    }
}

/// Cluster labels learned by [`DBSCAN`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedDBSCAN {
    labels: Array1<Option<usize>>,
    n_clusters: usize,
    n_features: usize,
}

impl FittedDBSCAN {
    /// Returns one label per training row, in training order.
    ///
    /// `Some(cluster)` names the zero-based cluster a row was assigned to;
    /// `None` marks a row as noise, reachable from no core point.
    #[must_use]
    pub const fn labels(&self) -> &Array1<Option<usize>> {
        &self.labels
    }

    /// Returns the number of clusters discovered.
    ///
    /// This does not count noise as a cluster.
    #[must_use]
    pub const fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of training rows labeled as noise.
    #[must_use]
    pub fn n_noise_points(&self) -> usize {
        self.labels.iter().filter(|label| label.is_none()).count()
    }
}

fn squared_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

fn validate_eps(eps: f64) -> Result<()> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(MlError::InvalidEps(eps));
    }
    Ok(())
}

fn validate_min_samples(min_samples: usize) -> Result<()> {
    if min_samples == 0 {
        return Err(MlError::InvalidMinSamples(min_samples));
    }
    Ok(())
}
