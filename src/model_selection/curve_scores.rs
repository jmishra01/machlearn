use ndarray::{Array1, Array2, Axis};

/// Train and test scores computed at every point along some varying axis,
/// one column per cross-validation fold.
///
/// [`super::learning_curve`] varies the training-set size;
/// [`super::validation_curve`] varies a hyperparameter value. Both report
/// their results in this shape so a widening gap between the train and
/// test means at a point is read the same way either time: a hint of
/// overfitting there.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CurveScores {
    train_scores: Array2<f64>,
    test_scores: Array2<f64>,
}

impl CurveScores {
    pub(super) const fn new(train_scores: Array2<f64>, test_scores: Array2<f64>) -> Self {
        Self {
            train_scores,
            test_scores,
        }
    }

    /// Returns training-fold scores, shaped `(n_points, n_folds)`.
    #[must_use]
    pub const fn train_scores(&self) -> &Array2<f64> {
        &self.train_scores
    }

    /// Returns test-fold scores, shaped `(n_points, n_folds)`.
    #[must_use]
    pub const fn test_scores(&self) -> &Array2<f64> {
        &self.test_scores
    }

    /// Returns the number of evaluated points.
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.train_scores.nrows()
    }

    /// Returns the number of cross-validation folds.
    #[must_use]
    pub fn n_folds(&self) -> usize {
        self.train_scores.ncols()
    }

    /// Returns each point's mean training-fold score across folds.
    #[must_use]
    pub fn train_scores_mean(&self) -> Array1<f64> {
        mean_across_folds(&self.train_scores)
    }

    /// Returns each point's mean test-fold score across folds.
    #[must_use]
    pub fn test_scores_mean(&self) -> Array1<f64> {
        mean_across_folds(&self.test_scores)
    }
}

fn mean_across_folds(scores: &Array2<f64>) -> Array1<f64> {
    #[allow(clippy::cast_precision_loss)]
    let n_folds = scores.ncols() as f64;
    scores.sum_axis(Axis(1)) / n_folds
}
