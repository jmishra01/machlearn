use std::cmp::Ordering;

use ndarray::{Array1, ArrayView1, ArrayView2};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::cross_validation::cross_validate_parallel_with_scorer;
use super::{
    CrossValidationScores, Fold, ParameterGrid, ParameterSet,
    cross_validation::{cross_validate_with_scorer, validate_folds},
};
use crate::core::{Dataset, Fit, Predict, Result};

/// Whether larger or smaller cross-validation scores are preferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScoreDirection {
    /// Rank larger mean scores first.
    Maximize,
    /// Rank smaller mean scores first.
    Minimize,
}

/// Cross-validation results for one concrete parameter assignment.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridSearchEntry {
    parameters: ParameterSet,
    scores: CrossValidationScores,
    rank: usize,
}

impl GridSearchEntry {
    /// Returns the evaluated hyperparameter assignment.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    /// Returns the individual fold scores.
    #[must_use]
    pub const fn scores(&self) -> &CrossValidationScores {
        &self.scores
    }

    /// Returns the arithmetic mean of the fold scores.
    #[must_use]
    pub fn mean_score(&self) -> f64 {
        self.scores.mean()
    }

    /// Returns the one-based competition rank.
    ///
    /// Entries with equal mean scores share a rank. A following distinct entry
    /// receives its one-based position, so ranks may contain gaps.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }
}

/// Deterministically ranked results from a parameter grid search.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridSearchResult {
    entries: Vec<GridSearchEntry>,
    direction: ScoreDirection,
}

impl GridSearchResult {
    /// Returns ranked entries, best first.
    #[must_use]
    pub fn entries(&self) -> &[GridSearchEntry] {
        &self.entries
    }

    /// Returns the best entry, if the result was deserialized from valid data.
    #[must_use]
    pub fn best(&self) -> Option<&GridSearchEntry> {
        self.entries.first()
    }

    /// Returns the number of evaluated parameter assignments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no entries are stored.
    ///
    /// Results produced by [`grid_search`] are never empty because an empty
    /// [`ParameterGrid`] represents one default configuration.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the score ordering used for ranking.
    #[must_use]
    pub const fn direction(&self) -> ScoreDirection {
        self.direction
    }
}

/// Evaluates and ranks every assignment in a hyperparameter grid.
///
/// `estimator_factory` builds a configured estimator from each parameter set.
/// Every estimator is then fitted independently for every fold through the
/// same isolation guarantees as [`super::cross_validate`]. Results are ranked
/// by mean score. Equal scores preserve deterministic parameter-grid order and
/// receive the same competition rank.
///
/// # Errors
///
/// Returns an error when grid expansion or estimator construction fails, folds
/// are invalid, fitting or prediction fails, or the scorer fails or returns a
/// non-finite value.
pub fn grid_search<Target, Factory, Estimator, Model, Prediction, Scorer>(
    grid: &ParameterGrid,
    estimator_factory: Factory,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: Scorer,
    direction: ScoreDirection,
) -> Result<GridSearchResult>
where
    Target: Clone,
    Factory: Fn(&ParameterSet) -> Result<Estimator>,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_folds(folds, dataset.n_samples())?;
    let parameter_sets = grid.combinations()?;
    let mut evaluated = Vec::new();
    evaluated
        .try_reserve_exact(parameter_sets.len())
        .map_err(|_| crate::core::MlError::ParameterGridTooLarge)?;
    for (candidate_index, parameters) in parameter_sets.into_iter().enumerate() {
        let estimator = estimator_factory(&parameters)?;
        let fold_scores = cross_validate_with_scorer(&estimator, dataset, folds, &scorer)?;
        evaluated.push((candidate_index, parameters, fold_scores));
    }

    rank_evaluated(evaluated, direction)
}

/// Evaluates parameter assignments and their folds in parallel.
///
/// This is the parallel counterpart to [`grid_search`]. Ranked entries, fold
/// scores, tie-breaking, and errors retain deterministic grid and fold order.
///
/// # Errors
///
/// Returns the same errors as [`grid_search`].
#[cfg(feature = "parallel")]
pub fn grid_search_parallel<Target, Factory, Estimator, Model, Prediction, Scorer>(
    grid: &ParameterGrid,
    estimator_factory: Factory,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: Scorer,
    direction: ScoreDirection,
) -> Result<GridSearchResult>
where
    Target: Clone + Sync,
    Factory: Fn(&ParameterSet) -> Result<Estimator> + Sync,
    Estimator: Sync,
    Scorer: Sync,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_folds(folds, dataset.n_samples())?;
    let parameter_sets = grid.combinations()?;
    let results: Vec<_> = parameter_sets
        .into_par_iter()
        .enumerate()
        .map(|(candidate_index, parameters)| {
            let estimator = estimator_factory(&parameters)?;
            let fold_scores =
                cross_validate_parallel_with_scorer(&estimator, dataset, folds, &scorer)?;
            Ok((candidate_index, parameters, fold_scores))
        })
        .collect();
    let evaluated = results.into_iter().collect::<Result<Vec<_>>>()?;
    rank_evaluated(evaluated, direction)
}

pub(super) type EvaluatedCandidate = (usize, ParameterSet, CrossValidationScores);

pub(super) fn rank_evaluated(
    mut evaluated: Vec<EvaluatedCandidate>,
    direction: ScoreDirection,
) -> Result<GridSearchResult> {
    evaluated.sort_by(|left, right| {
        compare_scores(direction, left.2.mean(), right.2.mean()).then_with(|| left.0.cmp(&right.0))
    });

    let mut entries = Vec::new();
    entries
        .try_reserve_exact(evaluated.len())
        .map_err(|_| crate::core::MlError::ParameterGridTooLarge)?;
    let mut previous_score = None;
    let mut rank = 0;
    for (position, (_candidate_index, parameters, fold_scores)) in evaluated.into_iter().enumerate()
    {
        let mean_score = fold_scores.mean();
        if previous_score.is_none_or(|score: f64| score.total_cmp(&mean_score) != Ordering::Equal) {
            rank = position + 1;
        }
        previous_score = Some(mean_score);
        entries.push(GridSearchEntry {
            parameters,
            scores: fold_scores,
            rank,
        });
    }

    Ok(GridSearchResult { entries, direction })
}

fn compare_scores(direction: ScoreDirection, left: f64, right: f64) -> Ordering {
    let ascending = left.total_cmp(&right);
    match direction {
        ScoreDirection::Minimize => ascending,
        ScoreDirection::Maximize => ascending.reverse(),
    }
}
