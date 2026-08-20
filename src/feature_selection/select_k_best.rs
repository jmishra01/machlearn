// `ArrayView1`/`ArrayView2` are lightweight view descriptors; accepting them
// by value avoids requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::core::{MlError, Result, Transform, validate_feature_count, validate_features};

/// Configures selection of the `k` highest-scoring features according to a
/// caller-supplied univariate scoring function, such as
/// [`crate::f_classif`] or [`crate::f_regression`].
///
/// Unlike [`crate::VarianceThreshold`], scoring looks at the relationship
/// between each feature and the training targets, not just the feature's
/// own spread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectKBest {
    k: usize,
}

impl SelectKBest {
    /// Creates a selector that keeps the `k` highest-scoring features.
    ///
    /// `k` may exceed the number of available features; in that case every
    /// feature is kept.
    #[must_use]
    pub const fn new(k: usize) -> Self {
        Self { k }
    }

    /// Returns the configured feature count.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }

    /// Scores every feature with `scorer` and selects the `k` highest
    /// scorers.
    ///
    /// Ties are broken in favor of the lowest column index, and a `NaN`
    /// score is always treated as the lowest possible score. Selected
    /// columns are kept in their original relative order.
    ///
    /// # Errors
    ///
    /// Returns an error when `records` and `targets` have different row
    /// counts, or when `scorer` fails.
    pub fn fit<Target, Scorer>(
        &self,
        records: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, Target>,
        scorer: Scorer,
    ) -> Result<FittedSelectKBest>
    where
        Scorer: Fn(ArrayView2<'_, f64>, ArrayView1<'_, Target>) -> Result<Array1<f64>>,
    {
        validate_features(records)?;
        if records.nrows() != targets.len() {
            return Err(MlError::MismatchedSampleCount {
                feature_rows: records.nrows(),
                target_count: targets.len(),
            });
        }

        let feature_scores = scorer(records, targets)?;
        let n_features = records.ncols();
        let selected_count = self.k.min(n_features);

        let mut order: Vec<usize> = (0..n_features).collect();
        order.sort_by(|&left, &right| {
            score_key(feature_scores[right])
                .partial_cmp(&score_key(feature_scores[left]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected_indices: Vec<usize> = order.into_iter().take(selected_count).collect();
        selected_indices.sort_unstable();

        Ok(FittedSelectKBest {
            scores: feature_scores,
            selected_indices,
            n_features,
        })
    }
}

/// Maps a score to a total order where `NaN` always sorts as the lowest
/// possible value, so a degenerate scorer output never wins selection.
fn score_key(score: f64) -> f64 {
    if score.is_nan() {
        f64::NEG_INFINITY
    } else {
        score
    }
}

/// The per-feature scores and selected columns learned by [`SelectKBest`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedSelectKBest {
    scores: Array1<f64>,
    selected_indices: Vec<usize>,
    n_features: usize,
}

impl FittedSelectKBest {
    /// Returns every input feature's score, in original column order.
    #[must_use]
    pub const fn scores(&self) -> &Array1<f64> {
        &self.scores
    }

    /// Returns the original column indices that were selected, in
    /// ascending order.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of features that were selected.
    #[must_use]
    pub fn n_selected_features(&self) -> usize {
        self.selected_indices.len()
    }

    /// Keeps only the selected columns, preserving their original relative
    /// order.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, or have the
    /// wrong column count.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;
        Ok(records.select(Axis(1), &self.selected_indices))
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedSelectKBest {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}
