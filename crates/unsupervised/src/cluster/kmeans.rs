// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{RngExt, SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;

use machlearn_core::core::{
    MlError, Predict, Result, Transform, validate_feature_count, validate_features,
};
use machlearn_preprocessing::preprocessing::{FittedTransformer, TransformerEstimator};

const DEFAULT_MAX_ITERATIONS: usize = 300;
const DEFAULT_TOLERANCE: f64 = 1.0e-4;
const DEFAULT_SEED: u64 = 42;

/// Determines how initial cluster centroids are chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KMeansInit {
    /// The k-means++ scheme: centroids are chosen one at a time, weighted by
    /// each point's squared distance to the closest centroid already chosen.
    /// This spreads initial centroids out and typically converges faster and
    /// more reliably than uniform random initialization.
    #[default]
    PlusPlus,
    /// `n_clusters` distinct training rows are chosen uniformly at random.
    Random,
}

/// Configures Lloyd's-algorithm k-means clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KMeans {
    n_clusters: usize,
    init: KMeansInit,
    max_iterations: usize,
    tolerance: f64,
    seed: u64,
}

impl KMeans {
    /// Creates a k-means estimator that partitions data into `n_clusters`
    /// clusters using k-means++ initialization.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_clusters` is zero.
    pub fn new(n_clusters: usize) -> Result<Self> {
        validate_n_clusters(n_clusters)?;
        Ok(Self {
            n_clusters,
            init: KMeansInit::PlusPlus,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            seed: DEFAULT_SEED,
        })
    }

    /// Sets how initial centroids are chosen.
    #[must_use]
    pub const fn with_init(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }

    /// Sets the maximum number of Lloyd's-algorithm iterations.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_iterations` is zero.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Result<Self> {
        validate_max_iterations(max_iterations)?;
        self.max_iterations = max_iterations;
        Ok(self)
    }

    /// Sets the centroid-shift tolerance that stops iteration early.
    ///
    /// Iteration stops once the summed squared movement of every centroid
    /// between iterations falls to or below this value.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is non-positive, NaN, or infinite.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self> {
        validate_tolerance(tolerance)?;
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Sets the deterministic seed used for centroid initialization.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the configured cluster count.
    #[must_use]
    pub const fn n_clusters(self) -> usize {
        self.n_clusters
    }

    /// Returns the configured initialization scheme.
    #[must_use]
    pub const fn init(self) -> KMeansInit {
        self.init
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

    /// Returns the configured random seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Partitions `records` into `n_clusters` clusters using Lloyd's
    /// algorithm.
    ///
    /// Iteration stops when centroid movement falls to or below `tolerance`,
    /// or after `max_iterations` iterations, whichever comes first; failing
    /// to converge within the iteration budget is not an error.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid cluster count, iteration budget, or
    /// tolerance, when features are empty or non-finite, or when there are
    /// fewer samples than clusters.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedKMeans> {
        validate_n_clusters(self.n_clusters)?;
        validate_max_iterations(self.max_iterations)?;
        validate_tolerance(self.tolerance)?;
        validate_features(records)?;
        if records.nrows() < self.n_clusters {
            return Err(MlError::InsufficientSamples {
                required: self.n_clusters,
                actual: records.nrows(),
            });
        }

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut centroids = match self.init {
            KMeansInit::PlusPlus => kmeans_plus_plus_init(records, self.n_clusters, &mut rng),
            KMeansInit::Random => random_init(records, self.n_clusters, &mut rng),
        };

        let mut assignments = assign_clusters(records, centroids.view());
        let mut iterations = 0;
        let mut converged = false;
        for _iteration in 0..self.max_iterations {
            iterations += 1;
            let updated = update_centroids(records, &assignments, &centroids);
            let shift = squared_centroid_shift(centroids.view(), updated.view());
            centroids = updated;
            assignments = assign_clusters(records, centroids.view());
            if shift <= self.tolerance {
                converged = true;
                break;
            }
        }

        let inertia = compute_inertia(records, centroids.view(), &assignments);

        Ok(FittedKMeans {
            centroids,
            iterations,
            converged,
            inertia,
        })
    }
}

/// Cluster centroids learned by [`KMeans`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedKMeans {
    centroids: Array2<f64>,
    iterations: usize,
    converged: bool,
    inertia: f64,
}

impl FittedKMeans {
    /// Returns one centroid row per cluster.
    #[must_use]
    pub const fn centroids(&self) -> &Array2<f64> {
        &self.centroids
    }

    /// Returns the number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.centroids.nrows()
    }

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.centroids.ncols()
    }

    /// Returns the number of Lloyd's-algorithm iterations completed.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns whether centroid movement fell to or below the configured
    /// tolerance before the iteration budget was exhausted.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the sum of squared distances from every training sample to
    /// its assigned centroid.
    ///
    /// Lower values indicate tighter clusters; this is the objective Lloyd's
    /// algorithm minimizes.
    #[must_use]
    pub const fn inertia(&self) -> f64 {
        self.inertia
    }

    /// Assigns every row to its nearest centroid.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn predict(&self, records: ArrayView2<'_, f64>) -> Result<Array1<usize>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;
        Ok(Array1::from_vec(assign_clusters(
            records,
            self.centroids.view(),
        )))
    }

    /// Computes the Euclidean distance from every row to every centroid.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features())?;
        let mut distances = Array2::zeros((records.nrows(), self.n_clusters()));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            for (cluster_index, centroid) in self.centroids.rows().into_iter().enumerate() {
                distances[[row_index, cluster_index]] = squared_distance(row, centroid).sqrt();
            }
        }
        Ok(distances)
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedKMeans {
    type Output = Array1<usize>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::predict(self, features)
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedKMeans {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

impl TransformerEstimator for KMeans {
    fn fit(&self, records: ArrayView2<'_, f64>) -> Result<Box<dyn FittedTransformer>> {
        Ok(Box::new(Self::fit(self, records)?))
    }
}

impl FittedTransformer for FittedKMeans {
    fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Self::transform(self, records)
    }
}

fn squared_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

fn nearest_centroid(row: ArrayView1<'_, f64>, centroids: ArrayView2<'_, f64>) -> usize {
    let mut best_cluster = 0;
    let mut best_distance = squared_distance(row, centroids.row(0));
    for (cluster_index, centroid) in centroids.rows().into_iter().enumerate().skip(1) {
        let distance = squared_distance(row, centroid);
        if distance < best_distance {
            best_distance = distance;
            best_cluster = cluster_index;
        }
    }
    best_cluster
}

fn assign_clusters(records: ArrayView2<'_, f64>, centroids: ArrayView2<'_, f64>) -> Vec<usize> {
    records
        .rows()
        .into_iter()
        .map(|row| nearest_centroid(row, centroids))
        .collect()
}

fn update_centroids(
    records: ArrayView2<'_, f64>,
    assignments: &[usize],
    previous_centroids: &Array2<f64>,
) -> Array2<f64> {
    let n_clusters = previous_centroids.nrows();
    let n_features = previous_centroids.ncols();
    let mut sums = Array2::zeros((n_clusters, n_features));
    let mut counts = vec![0usize; n_clusters];
    for (row, &cluster) in records.rows().into_iter().zip(assignments.iter()) {
        sums.row_mut(cluster).scaled_add(1.0, &row);
        counts[cluster] += 1;
    }

    let mut centroids = previous_centroids.clone();
    for (cluster, &count) in counts.iter().enumerate() {
        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let count = count as f64;
            centroids
                .row_mut(cluster)
                .assign(&(sums.row(cluster).to_owned() / count));
        }
    }
    centroids
}

fn squared_centroid_shift(previous: ArrayView2<'_, f64>, current: ArrayView2<'_, f64>) -> f64 {
    previous
        .rows()
        .into_iter()
        .zip(current.rows())
        .map(|(old_row, new_row)| squared_distance(old_row, new_row))
        .sum()
}

fn compute_inertia(
    records: ArrayView2<'_, f64>,
    centroids: ArrayView2<'_, f64>,
    assignments: &[usize],
) -> f64 {
    records
        .rows()
        .into_iter()
        .zip(assignments.iter())
        .map(|(row, &cluster)| squared_distance(row, centroids.row(cluster)))
        .sum()
}

fn random_init(
    records: ArrayView2<'_, f64>,
    n_clusters: usize,
    rng: &mut ChaCha8Rng,
) -> Array2<f64> {
    let indices = index::sample(rng, records.nrows(), n_clusters).into_vec();
    records.select(Axis(0), &indices)
}

fn kmeans_plus_plus_init(
    records: ArrayView2<'_, f64>,
    n_clusters: usize,
    rng: &mut ChaCha8Rng,
) -> Array2<f64> {
    let n_samples = records.nrows();
    let n_features = records.ncols();
    let mut centroids = Array2::zeros((n_clusters, n_features));

    let first_index = rng.random_range(0..n_samples);
    centroids.row_mut(0).assign(&records.row(first_index));

    let mut closest_squared_distances: Vec<f64> = records
        .rows()
        .into_iter()
        .map(|row| squared_distance(row, centroids.row(0)))
        .collect();

    for cluster_index in 1..n_clusters {
        let total: f64 = closest_squared_distances.iter().sum();
        let chosen_index = if total > 0.0 {
            let threshold = rng.random_range(0.0..total);
            let mut cumulative = 0.0;
            let mut chosen = n_samples - 1;
            for (row_index, &distance) in closest_squared_distances.iter().enumerate() {
                cumulative += distance;
                if cumulative >= threshold {
                    chosen = row_index;
                    break;
                }
            }
            chosen
        } else {
            rng.random_range(0..n_samples)
        };

        centroids
            .row_mut(cluster_index)
            .assign(&records.row(chosen_index));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            let distance = squared_distance(row, centroids.row(cluster_index));
            if distance < closest_squared_distances[row_index] {
                closest_squared_distances[row_index] = distance;
            }
        }
    }

    centroids
}

fn validate_n_clusters(n_clusters: usize) -> Result<()> {
    if n_clusters == 0 {
        return Err(MlError::InvalidClusterCount(n_clusters));
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
