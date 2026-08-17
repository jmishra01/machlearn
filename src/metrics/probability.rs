// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use ndarray::ArrayView1;

use crate::core::{MlError, Result};

/// Returns binary cross-entropy for positive-class probabilities.
///
/// Exact zero and one probabilities are clipped to `f64::EPSILON` and
/// `1 - f64::EPSILON`, respectively, so the result remains finite. Values
/// outside `[0, 1]` are rejected rather than clipped.
///
/// # Errors
///
/// Returns an error for empty or different-length inputs, non-finite or
/// out-of-range probabilities, a target collection without exactly two
/// observed classes, or a positive label absent from the targets.
pub fn binary_log_loss<Label>(
    actual: ArrayView1<'_, Label>,
    positive_probabilities: ArrayView1<'_, f64>,
    positive_label: &Label,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    validate_binary_inputs(actual, positive_probabilities, positive_label)?;
    #[allow(clippy::cast_precision_loss)]
    let count = actual.len() as f64;
    let upper = 1.0 - f64::EPSILON;
    let loss = actual
        .iter()
        .zip(positive_probabilities)
        .map(|(label, &probability)| {
            let probability = probability.clamp(f64::EPSILON, upper);
            if label == positive_label {
                -probability.ln() / count
            } else {
                -(1.0 - probability).ln() / count
            }
        })
        .sum();
    finite_result("binary_log_loss", loss)
}

/// Returns the area under the binary receiver-operating-characteristic curve.
///
/// This is the probability that a randomly selected positive observation has a
/// higher score than a randomly selected negative observation. Tied scores
/// receive half credit through average ranks.
///
/// # Errors
///
/// Returns an error for empty or different-length inputs, non-finite or
/// out-of-range probabilities, a target collection without exactly two
/// observed classes, or a positive label absent from the targets.
#[allow(clippy::float_cmp)]
pub fn roc_auc_score<Label>(
    actual: ArrayView1<'_, Label>,
    positive_probabilities: ArrayView1<'_, f64>,
    positive_label: &Label,
) -> Result<f64>
where
    Label: Clone + Ord,
{
    validate_binary_inputs(actual, positive_probabilities, positive_label)?;

    let mut ranked: Vec<_> = positive_probabilities
        .iter()
        .copied()
        .zip(actual.iter().map(|label| label == positive_label))
        .collect();
    ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    let positive_count = ranked.iter().filter(|(_, positive)| *positive).count();
    let negative_count = ranked.len() - positive_count;
    let mut positive_rank_sum = 0.0;
    let mut start = 0;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len() && ranked[end].0 == ranked[start].0 {
            end += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let average_rank = f64::midpoint((start + 1) as f64, end as f64);
        let positives_in_group = ranked[start..end]
            .iter()
            .filter(|(_, positive)| *positive)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let positives_in_group = positives_in_group as f64;
        positive_rank_sum += average_rank * positives_in_group;
        start = end;
    }

    #[allow(clippy::cast_precision_loss)]
    let positive_count = positive_count as f64;
    #[allow(clippy::cast_precision_loss)]
    let negative_count = negative_count as f64;
    let auc = (positive_rank_sum - positive_count * (positive_count + 1.0) / 2.0)
        / (positive_count * negative_count);
    finite_result("roc_auc_score", auc)
}

fn validate_binary_inputs<Label>(
    actual: ArrayView1<'_, Label>,
    probabilities: ArrayView1<'_, f64>,
    positive_label: &Label,
) -> Result<()>
where
    Label: Clone + Ord,
{
    if actual.is_empty() {
        return Err(MlError::EmptyMetricInput);
    }
    if actual.len() != probabilities.len() {
        return Err(MlError::MismatchedMetricInput {
            actual: actual.len(),
            predicted: probabilities.len(),
        });
    }
    for (index, &probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() {
            return Err(MlError::NonFiniteProbability { index });
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(MlError::InvalidProbability {
                index,
                value: probability,
            });
        }
    }

    let mut classes = actual.to_vec();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() != 2 {
        return Err(MlError::ExpectedBinaryTargets {
            class_count: classes.len(),
        });
    }
    if classes.binary_search(positive_label).is_err() {
        return Err(MlError::PositiveLabelNotFound);
    }
    Ok(())
}

fn finite_result(metric: &'static str, result: f64) -> Result<f64> {
    if !result.is_finite() {
        return Err(MlError::NonFiniteMetricResult { metric });
    }
    Ok(result)
}
