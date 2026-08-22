// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use std::cmp::Ordering;

use ndarray::{Array1, ArrayView1, ArrayView2};

use machlearn_core::core::{MlError, Result};

/// A binary decision tree over `f64` features, generic over the payload
/// stored at each leaf (class probabilities for a classifier, a mean value
/// for a regressor).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum Node<Leaf> {
    Leaf(Leaf),
    Split {
        feature_index: usize,
        threshold: f64,
        /// Training rows that reached this node, used to weight this split's
        /// contribution to feature-importance reporting.
        n_samples: usize,
        /// Node impurity minus the sample-size-weighted impurity of the two
        /// children, i.e. how much this split reduced impurity.
        impurity_decrease: f64,
        left: Box<Node<Leaf>>,
        right: Box<Node<Leaf>>,
    },
}

impl<Leaf> Node<Leaf> {
    /// Walks the tree for `row`, following the left branch when the split
    /// feature is at most the threshold and the right branch otherwise.
    pub(crate) fn leaf_for(&self, row: ArrayView1<'_, f64>) -> &Leaf {
        match self {
            Self::Leaf(leaf) => leaf,
            Self::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                if row[*feature_index] <= *threshold {
                    left.leaf_for(row)
                } else {
                    right.leaf_for(row)
                }
            }
        }
    }

    fn accumulate_importances(&self, importances: &mut [f64]) {
        if let Self::Split {
            feature_index,
            n_samples,
            impurity_decrease,
            left,
            right,
            ..
        } = self
        {
            #[allow(clippy::cast_precision_loss)]
            let weight = *n_samples as f64;
            importances[*feature_index] += weight * impurity_decrease;
            left.accumulate_importances(importances);
            right.accumulate_importances(importances);
        }
    }
}

/// Mean-decrease-in-impurity feature importances for a single tree,
/// normalized to sum to one.
///
/// A tree with no splits (a single leaf) has no informative feature and
/// reports every importance as zero rather than dividing by zero.
pub(crate) fn feature_importances<Leaf>(root: &Node<Leaf>, n_features: usize) -> Array1<f64> {
    let mut importances = vec![0.0_f64; n_features];
    root.accumulate_importances(&mut importances);
    let total: f64 = importances.iter().sum();
    if total > 0.0 {
        for value in &mut importances {
            *value /= total;
        }
    }
    Array1::from_vec(importances)
}

/// Depth and sample-count constraints that stop tree growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GrowthLimits {
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
}

struct SplitCandidate {
    feature_index: usize,
    threshold: f64,
    /// Sample-size-weighted average impurity of the two children, i.e. the
    /// score this split was chosen to minimize.
    weighted_impurity: f64,
    left_rows: Vec<usize>,
    right_rows: Vec<usize>,
}

/// Finds the lowest-weighted-impurity binary split among the midpoints
/// between consecutive distinct sorted values of every candidate feature,
/// restricted to splits where both partitions have at least
/// `min_samples_leaf` rows.
///
/// Every row carries the weight recorded at its index in `weights`, which
/// determines how much each partition's impurity contributes to the split
/// score; `min_samples_leaf` is always a plain row count, unaffected by
/// weighting. Ties are resolved in favor of the first candidate feature and,
/// within a feature, the smallest threshold, making the search deterministic
/// for a given candidate feature order.
fn best_split<Impurity>(
    records: ArrayView2<'_, f64>,
    rows: &[usize],
    candidate_features: &[usize],
    min_samples_leaf: usize,
    weights: ArrayView1<'_, f64>,
    impurity: &Impurity,
) -> Option<SplitCandidate>
where
    Impurity: Fn(&[usize]) -> f64,
{
    let total: f64 = rows.iter().map(|&row| weights[row]).sum();
    let mut best_score = f64::INFINITY;
    let mut best = None;

    for &feature_index in candidate_features {
        let mut ordered = rows.to_vec();
        ordered.sort_by(|&left, &right| {
            records[[left, feature_index]].total_cmp(&records[[right, feature_index]])
        });

        for split_at in 1..ordered.len() {
            let previous_value = records[[ordered[split_at - 1], feature_index]];
            let current_value = records[[ordered[split_at], feature_index]];
            if previous_value.total_cmp(&current_value) == Ordering::Equal {
                continue;
            }

            let left_rows = &ordered[..split_at];
            let right_rows = &ordered[split_at..];
            if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
                continue;
            }

            let left_weight: f64 = left_rows.iter().map(|&row| weights[row]).sum();
            let right_weight: f64 = right_rows.iter().map(|&row| weights[row]).sum();
            let weighted_impurity =
                (left_weight * impurity(left_rows) + right_weight * impurity(right_rows)) / total;

            if weighted_impurity < best_score {
                best_score = weighted_impurity;
                best = Some(SplitCandidate {
                    feature_index,
                    threshold: f64::midpoint(previous_value, current_value),
                    weighted_impurity,
                    left_rows: left_rows.to_vec(),
                    right_rows: right_rows.to_vec(),
                });
            }
        }
    }

    best
}

/// Recursively grows a binary decision tree.
///
/// `feature_sampler` is invoked with the total feature count immediately
/// before searching for each split and returns the candidate feature indices
/// to consider; a plain decision tree offers every feature at every split,
/// while a random-forest member tree offers a freshly drawn random subset.
///
/// Splitting stops when the node is pure (`impurity` is `0`), the sample
/// count is below `min_samples_split`, the depth budget is exhausted, or no
/// split respects `min_samples_leaf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_tree<Leaf, Impurity, MakeLeaf, FeatureSampler>(
    records: ArrayView2<'_, f64>,
    rows: Vec<usize>,
    depth: usize,
    limits: &GrowthLimits,
    weights: ArrayView1<'_, f64>,
    impurity: &Impurity,
    make_leaf: &MakeLeaf,
    feature_sampler: &mut FeatureSampler,
) -> Node<Leaf>
where
    Impurity: Fn(&[usize]) -> f64,
    MakeLeaf: Fn(&[usize]) -> Leaf,
    FeatureSampler: FnMut(usize) -> Vec<usize>,
{
    let depth_exhausted = limits.max_depth.is_some_and(|max| depth >= max);
    let node_impurity = impurity(&rows);
    let splittable =
        rows.len() >= limits.min_samples_split && !depth_exhausted && node_impurity > 0.0;

    if splittable {
        let n_samples = rows.len();
        let candidate_features = feature_sampler(records.ncols());
        if let Some(split) = best_split(
            records,
            &rows,
            &candidate_features,
            limits.min_samples_leaf,
            weights,
            impurity,
        ) {
            let left = build_tree(
                records,
                split.left_rows,
                depth + 1,
                limits,
                weights,
                impurity,
                make_leaf,
                feature_sampler,
            );
            let right = build_tree(
                records,
                split.right_rows,
                depth + 1,
                limits,
                weights,
                impurity,
                make_leaf,
                feature_sampler,
            );
            return Node::Split {
                feature_index: split.feature_index,
                threshold: split.threshold,
                n_samples,
                impurity_decrease: node_impurity - split.weighted_impurity,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    Node::Leaf(make_leaf(&rows))
}

pub(crate) fn validate_min_samples_split(min_samples_split: usize) -> Result<()> {
    if min_samples_split < 2 {
        return Err(MlError::InvalidMinSamplesSplit(min_samples_split));
    }
    Ok(())
}

pub(crate) fn validate_min_samples_leaf(min_samples_leaf: usize) -> Result<()> {
    if min_samples_leaf == 0 {
        return Err(MlError::InvalidMinSamplesLeaf(min_samples_leaf));
    }
    Ok(())
}

/// Gini impurity of the sample-weighted class distribution among `rows`.
///
/// Every row carries the weight recorded at its index in `weights`; passing
/// a uniform weight of one for every row reduces this to plain (unweighted)
/// Gini impurity.
pub(crate) fn gini_impurity(
    encoded: &Array1<usize>,
    n_classes: usize,
    weights: ArrayView1<'_, f64>,
    rows: &[usize],
) -> f64 {
    let mut totals = vec![0.0_f64; n_classes];
    let mut total = 0.0;
    for &row in rows {
        totals[encoded[row]] += weights[row];
        total += weights[row];
    }
    1.0 - totals
        .iter()
        .map(|&weight| {
            let fraction = weight / total;
            fraction * fraction
        })
        .sum::<f64>()
}

/// The sample-weighted class-frequency distribution among `rows`, in
/// class-index order.
///
/// Every row carries the weight recorded at its index in `weights`; passing
/// a uniform weight of one for every row reduces this to plain (unweighted)
/// class frequencies.
pub(crate) fn leaf_probabilities(
    encoded: &Array1<usize>,
    n_classes: usize,
    weights: ArrayView1<'_, f64>,
    rows: &[usize],
) -> Array1<f64> {
    let mut totals = vec![0.0_f64; n_classes];
    let mut total = 0.0;
    for &row in rows {
        totals[encoded[row]] += weights[row];
        total += weights[row];
    }
    Array1::from_vec(totals.into_iter().map(|weight| weight / total).collect())
}

/// Population variance of the targets at `rows`.
pub(crate) fn target_variance(targets: &Array1<f64>, rows: &[usize]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let total = rows.len() as f64;
    let mean_value = rows.iter().map(|&row| targets[row]).sum::<f64>() / total;
    rows.iter()
        .map(|&row| (targets[row] - mean_value).powi(2))
        .sum::<f64>()
        / total
}

/// Arithmetic mean of the targets at `rows`.
pub(crate) fn target_mean(targets: &Array1<f64>, rows: &[usize]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let total = rows.len() as f64;
    rows.iter().map(|&row| targets[row]).sum::<f64>() / total
}
