// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{ArrayView1, ArrayView2};

use machlearn_core::core::{MlError, Result};

const MINIMUM_DISTANCE: f64 = 1.0e-12;

/// Determines how a neighbor's target contributes to a prediction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Weighting {
    /// Every neighbor contributes equally.
    #[default]
    Uniform,
    /// Closer neighbors contribute more, weighted by `1 / distance`.
    ///
    /// An exact match (distance zero) receives the entire weight so the
    /// contribution never divides by zero.
    Distance,
}

pub(super) fn validate_n_neighbors(n_neighbors: usize) -> Result<()> {
    if n_neighbors == 0 {
        return Err(MlError::InvalidNeighborCount(n_neighbors));
    }
    Ok(())
}

pub(super) fn validate_training_size(n_neighbors: usize, n_samples: usize) -> Result<()> {
    if n_samples < n_neighbors {
        return Err(MlError::InsufficientSamples {
            required: n_neighbors,
            actual: n_samples,
        });
    }
    Ok(())
}

fn euclidean_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Returns the `n_neighbors` training rows closest to `query`, paired with
/// their Euclidean distance.
///
/// Results are ordered by ascending distance, with ties broken by ascending
/// training-row index so results are deterministic regardless of row order.
pub(super) fn nearest_neighbors(
    training_records: ArrayView2<'_, f64>,
    query: ArrayView1<'_, f64>,
    n_neighbors: usize,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = training_records
        .rows()
        .into_iter()
        .enumerate()
        .map(|(index, row)| (index, euclidean_distance(row, query)))
        .collect();
    distances.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    distances.truncate(n_neighbors);
    distances
}

/// Converts neighbor distances into non-negative contribution weights.
pub(super) fn neighbor_weights(weighting: Weighting, distances: &[(usize, f64)]) -> Vec<f64> {
    match weighting {
        Weighting::Uniform => vec![1.0; distances.len()],
        Weighting::Distance => {
            let exact_match = distances
                .iter()
                .position(|&(_, distance)| distance <= MINIMUM_DISTANCE);
            exact_match.map_or_else(
                || {
                    distances
                        .iter()
                        .map(|&(_, distance)| 1.0 / distance)
                        .collect()
                },
                |exact_index| {
                    let mut weights = vec![0.0; distances.len()];
                    weights[exact_index] = 1.0;
                    weights
                },
            )
        }
    }
}
