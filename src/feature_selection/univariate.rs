// `ArrayView1`/`ArrayView2` are lightweight view descriptors; accepting them
// by value avoids requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::core::{MlError, Result, validate_features};

/// Scores every feature by its one-way ANOVA F-statistic across classes.
///
/// A large F-statistic means a feature's mean differs sharply between
/// classes relative to its spread within a class, so it is potentially
/// informative for classification. Suitable as a scorer for
/// [`crate::SelectKBest`] over classification targets.
///
/// # Errors
///
/// Returns an error when features are empty or non-finite, when `targets`
/// has a different row count than `records`, or when fewer than two
/// distinct classes are observed.
pub fn f_classif<Label>(
    records: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, Label>,
) -> Result<Array1<f64>>
where
    Label: Clone + Ord,
{
    validate_features(records)?;
    if records.nrows() != targets.len() {
        return Err(MlError::MismatchedSampleCount {
            feature_rows: records.nrows(),
            target_count: targets.len(),
        });
    }

    let mut classes = targets.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();
    if n_classes < 2 {
        return Err(MlError::InsufficientClasses {
            required: 2,
            actual: n_classes,
        });
    }

    let class_index_per_row: Vec<usize> = targets
        .iter()
        .map(|label| classes.binary_search(label).unwrap_or(0))
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n_samples = records.nrows() as f64;
    #[allow(clippy::cast_precision_loss)]
    let degrees_of_freedom_between = (n_classes - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let degrees_of_freedom_within = (records.nrows() - n_classes) as f64;

    Ok(Array1::from_iter(records.axis_iter(Axis(1)).map(
        |column| {
            let mut group_sums = vec![0.0_f64; n_classes];
            let mut group_counts = vec![0.0_f64; n_classes];
            for (row, &value) in column.iter().enumerate() {
                let class_index = class_index_per_row[row];
                group_sums[class_index] += value;
                group_counts[class_index] += 1.0;
            }
            let overall_mean = column.sum() / n_samples;
            let group_means: Vec<f64> = group_sums
                .iter()
                .zip(&group_counts)
                .map(|(&sum, &count)| sum / count)
                .collect();

            let between_group_sum_of_squares: f64 = group_counts
                .iter()
                .zip(&group_means)
                .map(|(&count, &mean)| count * (mean - overall_mean).powi(2))
                .sum();
            let within_group_sum_of_squares: f64 = column
                .iter()
                .enumerate()
                .map(|(row, &value)| (value - group_means[class_index_per_row[row]]).powi(2))
                .sum();

            (between_group_sum_of_squares / degrees_of_freedom_between)
                / (within_group_sum_of_squares / degrees_of_freedom_within)
        },
    )))
}

/// Scores every feature by an F-statistic derived from its Pearson
/// correlation with a continuous target.
///
/// A large F-statistic means a feature is strongly (positively or
/// negatively) correlated with the target, so it is potentially
/// informative for regression. Suitable as a scorer for
/// [`crate::SelectKBest`] over regression targets.
///
/// # Errors
///
/// Returns an error when features are empty or non-finite, or when
/// `targets` has a different row count than `records`.
pub fn f_regression(
    records: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Result<Array1<f64>> {
    validate_features(records)?;
    if records.nrows() != targets.len() {
        return Err(MlError::MismatchedSampleCount {
            feature_rows: records.nrows(),
            target_count: targets.len(),
        });
    }

    #[allow(clippy::cast_precision_loss)]
    let n_samples = records.nrows() as f64;
    let target_mean = targets.sum() / n_samples;
    let target_deviation: Array1<f64> = targets.mapv(|value| value - target_mean);
    let target_variance: f64 = target_deviation.iter().map(|value| value * value).sum();

    Ok(Array1::from_iter(records.axis_iter(Axis(1)).map(
        |column| {
            let feature_mean = column.sum() / n_samples;
            let mut covariance = 0.0;
            let mut feature_variance = 0.0;
            for (row, &value) in column.iter().enumerate() {
                let feature_deviation = value - feature_mean;
                covariance += feature_deviation * target_deviation[row];
                feature_variance += feature_deviation * feature_deviation;
            }
            let r_squared = (covariance * covariance) / (feature_variance * target_variance);
            r_squared * (n_samples - 2.0) / (1.0 - r_squared)
        },
    )))
}
