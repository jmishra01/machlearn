use ndarray::{Array1, ArrayView1, ArrayView2};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::cross_validation::cross_validate_parallel_with_scorer;
use super::{
    Fold, GridSearchResult, ParameterGrid, ParameterSet, ScoreDirection,
    cross_validation::{cross_validate_with_scorer, validate_folds},
    grid_search::rank_evaluated,
};
use machlearn_core::core::{Dataset, Fit, Predict, Result};

/// Evaluates and ranks `n_iter` randomly drawn assignments from a
/// hyperparameter grid.
///
/// Unlike [`super::grid_search`], which exhaustively evaluates every
/// combination, this draws `n_iter` assignments via
/// [`ParameterGrid::sample_combinations`], choosing each parameter's value
/// independently and uniformly from its candidates on every draw. This
/// trades an exhaustive search for a bounded evaluation budget, letting a
/// grid with far too many combinations to enumerate still be searched.
/// `estimator_factory` builds a configured estimator from each drawn
/// parameter set; every estimator is then fitted independently for every
/// fold through the same isolation guarantees as [`super::cross_validate`].
/// Results are ranked by mean score.
///
/// # Errors
///
/// Returns an error when `n_iter` is zero, folds are invalid, fitting or
/// prediction fails, or the scorer fails or returns a non-finite value.
#[allow(clippy::too_many_arguments)]
pub fn randomized_search<Target, Factory, Estimator, Model, Prediction, Scorer>(
    grid: &ParameterGrid,
    n_iter: usize,
    seed: u64,
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
    let parameter_sets = grid.sample_combinations(n_iter, seed)?;
    let mut evaluated = Vec::with_capacity(parameter_sets.len());
    for (candidate_index, parameters) in parameter_sets.into_iter().enumerate() {
        let estimator = estimator_factory(&parameters)?;
        let fold_scores = cross_validate_with_scorer(&estimator, dataset, folds, &scorer)?;
        evaluated.push((candidate_index, parameters, fold_scores));
    }

    rank_evaluated(evaluated, direction)
}

/// Evaluates randomly drawn parameter assignments and their folds in
/// parallel.
///
/// This is the parallel counterpart to [`randomized_search`]. Ranked
/// entries, fold scores, tie-breaking, and errors retain deterministic
/// draw and fold order.
///
/// # Errors
///
/// Returns the same errors as [`randomized_search`].
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn randomized_search_parallel<Target, Factory, Estimator, Model, Prediction, Scorer>(
    grid: &ParameterGrid,
    n_iter: usize,
    seed: u64,
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
    let parameter_sets = grid.sample_combinations(n_iter, seed)?;
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
