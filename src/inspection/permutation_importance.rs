use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{SeedableRng, seq::index};
use rand_chacha::ChaCha8Rng;

use crate::{
    core::{Dataset, MlError, Predict, Result},
    model_selection::ScoreDirection,
};

/// Per-feature, per-repeat score movements produced by
/// [`permutation_importance`].
///
/// A feature's importance for a repeat is how much worse the model's score
/// got after randomly shuffling that feature's values across rows, holding
/// every other feature fixed, oriented so a positive value always means
/// "worse" regardless of the scorer's [`ScoreDirection`]. A large,
/// consistently positive value means the model relies heavily on that
/// feature; a value near zero (or negative, from noise) means it does not.
/// Unlike a tree ensemble's built-in impurity-based feature importances,
/// this technique works with any fitted model and any scoring function.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PermutationImportance {
    importances: Array2<f64>,
}

impl PermutationImportance {
    /// Returns the raw score decreases, shaped `(n_features, n_repeats)`.
    #[must_use]
    pub fn importances(&self) -> ArrayView2<'_, f64> {
        self.importances.view()
    }

    /// Returns the number of features scored.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.importances.nrows()
    }

    /// Returns the number of shuffles performed per feature.
    #[must_use]
    pub fn n_repeats(&self) -> usize {
        self.importances.ncols()
    }

    /// Returns each feature's mean score decrease across repeats, in
    /// feature-column order.
    #[must_use]
    pub fn importances_mean(&self) -> Array1<f64> {
        #[allow(clippy::cast_precision_loss)]
        let count = self.importances.ncols() as f64;
        self.importances
            .rows()
            .into_iter()
            .map(|row| row.sum() / count)
            .collect()
    }

    /// Returns each feature's population standard deviation of score
    /// decreases across repeats, in feature-column order.
    #[must_use]
    pub fn importances_std(&self) -> Array1<f64> {
        let means = self.importances_mean();
        #[allow(clippy::cast_precision_loss)]
        let count = self.importances.ncols() as f64;
        Array1::from_iter(
            self.importances
                .rows()
                .into_iter()
                .enumerate()
                .map(|(feature, row)| {
                    let mean = means[feature];
                    (row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / count).sqrt()
                }),
        )
    }
}

/// Measures how much a fitted model's score depends on each feature by
/// repeatedly shuffling that feature's values and re-scoring.
///
/// For every feature column, `n_repeats` independent random permutations of
/// that column (with every other column held fixed) are scored against the
/// model's baseline (unpermuted) score; how far each repeat's score moves
/// away from baseline *in the direction that means the feature mattered* is
/// that repeat's importance. `direction` names which way a better score
/// moves for `scorer`, exactly as in [`crate::grid_search`]: pass
/// [`ScoreDirection::Maximize`] for a score like accuracy or R-squared, or
/// [`ScoreDirection::Minimize`] for an error like mean squared error, so
/// that a feature the model depends on always reports a positive
/// importance under either convention. Because it only needs [`Predict`]
/// and a scoring function, this works with any already-fitted model, not
/// just tree ensembles with a built-in impurity-based feature importance.
///
/// # Errors
///
/// Returns an error when `n_repeats` is zero, or when predicting or scoring
/// the baseline or any permuted feature matrix fails.
pub fn permutation_importance<Model, Target, Prediction, Scorer>(
    model: &Model,
    dataset: &Dataset<Target>,
    scorer: Scorer,
    direction: ScoreDirection,
    n_repeats: usize,
    seed: u64,
) -> Result<PermutationImportance>
where
    Target: Clone,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_n_repeats(n_repeats)?;

    let records = dataset.records();
    let targets = dataset.targets();
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();

    let baseline_predictions = model.predict(records)?;
    let baseline_score = scorer(targets, baseline_predictions.view())?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut permuted_records = records.to_owned();
    let mut importances = Array2::zeros((n_features, n_repeats));

    for feature in 0..n_features {
        let original_column: Array1<f64> = records.column(feature).to_owned();

        for repeat in 0..n_repeats {
            let permutation = index::sample(&mut rng, n_samples, n_samples).into_vec();
            for (row, &source_row) in permutation.iter().enumerate() {
                permuted_records[[row, feature]] = original_column[source_row];
            }

            let predictions = model.predict(permuted_records.view())?;
            let score = scorer(targets, predictions.view())?;
            importances[[feature, repeat]] = match direction {
                ScoreDirection::Maximize => baseline_score - score,
                ScoreDirection::Minimize => score - baseline_score,
            };
        }

        permuted_records
            .column_mut(feature)
            .assign(&original_column);
    }

    Ok(PermutationImportance { importances })
}

fn validate_n_repeats(n_repeats: usize) -> Result<()> {
    if n_repeats == 0 {
        return Err(MlError::InvalidRepeatCount(n_repeats));
    }
    Ok(())
}
