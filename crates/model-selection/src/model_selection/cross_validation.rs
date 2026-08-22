use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::Fold;
use machlearn_core::core::{Dataset, Fit, MlError, Predict, Result};

/// Scores produced by evaluating an estimator across multiple folds.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CrossValidationScores {
    scores: Vec<f64>,
}

impl CrossValidationScores {
    /// Returns scores in the same order as the supplied folds.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Returns the arithmetic mean of all fold scores.
    #[must_use]
    pub fn mean(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let count = self.scores.len() as f64;
        self.scores.iter().map(|score| score / count).sum()
    }

    /// Returns the population standard deviation of the fold scores.
    #[must_use]
    pub fn standard_deviation(&self) -> f64 {
        let mean = self.mean();
        #[allow(clippy::cast_precision_loss)]
        let count = self.scores.len() as f64;
        let variance = self
            .scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / count;
        variance.sqrt()
    }
}

/// Fits and scores an estimator independently for every supplied fold.
///
/// Each fit receives a newly selected training [`Dataset`]. Predictions are
/// generated only for that fold's test features, preventing estimator or
/// preprocessing state from leaking between folds. Scores retain fold order.
///
/// # Errors
///
/// Returns an error when fewer than two folds are supplied, a fold is not a
/// complete and disjoint partition of the dataset, fitting or prediction
/// fails, the scorer fails, or a scorer returns NaN or infinity.
pub fn cross_validate<Target, Estimator, Model, Prediction, Scorer>(
    estimator: &Estimator,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: Scorer,
) -> Result<CrossValidationScores>
where
    Target: Clone,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    cross_validate_with_scorer(estimator, dataset, folds, &scorer)
}

/// Fits and scores an estimator across folds in parallel.
///
/// This is the parallel counterpart to [`cross_validate`]. Fold scores and
/// errors retain input-fold order regardless of execution order.
///
/// # Errors
///
/// Returns the same errors as [`cross_validate`].
#[cfg(feature = "parallel")]
pub fn cross_validate_parallel<Target, Estimator, Model, Prediction, Scorer>(
    estimator: &Estimator,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: Scorer,
) -> Result<CrossValidationScores>
where
    Target: Clone + Sync,
    Estimator: Sync,
    Scorer: Sync,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    cross_validate_parallel_with_scorer(estimator, dataset, folds, &scorer)
}

pub(super) fn cross_validate_with_scorer<Target, Estimator, Model, Prediction, Scorer>(
    estimator: &Estimator,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: &Scorer,
) -> Result<CrossValidationScores>
where
    Target: Clone,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_folds(folds, dataset.n_samples())?;

    let mut fold_scores = Vec::with_capacity(folds.len());
    for (fold_index, fold) in folds.iter().enumerate() {
        fold_scores.push(evaluate_fold(estimator, dataset, fold, fold_index, scorer)?);
    }
    Ok(CrossValidationScores {
        scores: fold_scores,
    })
}

#[cfg(feature = "parallel")]
pub(super) fn cross_validate_parallel_with_scorer<Target, Estimator, Model, Prediction, Scorer>(
    estimator: &Estimator,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: &Scorer,
) -> Result<CrossValidationScores>
where
    Target: Clone + Sync,
    Estimator: Sync,
    Scorer: Sync,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_folds(folds, dataset.n_samples())?;
    let results: Vec<_> = folds
        .par_iter()
        .enumerate()
        .map(|(fold_index, fold)| evaluate_fold(estimator, dataset, fold, fold_index, scorer))
        .collect();
    let fold_scores = results.into_iter().collect::<Result<Vec<_>>>()?;
    Ok(CrossValidationScores {
        scores: fold_scores,
    })
}

pub(super) fn validate_folds(folds: &[Fold], n_samples: usize) -> Result<()> {
    if folds.len() < 2 {
        return Err(MlError::InvalidFoldCount {
            n_splits: folds.len(),
        });
    }
    for (fold_index, fold) in folds.iter().enumerate() {
        validate_fold(fold, fold_index, n_samples)?;
    }
    Ok(())
}

fn validate_fold(fold: &Fold, fold_index: usize, n_samples: usize) -> Result<()> {
    if fold.train_indices().is_empty()
        || fold.test_indices().is_empty()
        || fold.train_size() + fold.test_size() != n_samples
    {
        return Err(MlError::InvalidFoldPartition { fold_index });
    }

    let mut observed = vec![false; n_samples];
    for &sample_index in fold.train_indices().iter().chain(fold.test_indices()) {
        let Some(was_observed) = observed.get_mut(sample_index) else {
            return Err(MlError::InvalidFoldPartition { fold_index });
        };
        if *was_observed {
            return Err(MlError::InvalidFoldPartition { fold_index });
        }
        *was_observed = true;
    }
    Ok(())
}

fn evaluate_fold<Target, Estimator, Model, Prediction, Scorer>(
    estimator: &Estimator,
    dataset: &Dataset<Target>,
    fold: &Fold,
    fold_index: usize,
    scorer: &Scorer,
) -> Result<f64>
where
    Target: Clone,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    let training = select(dataset, fold.train_indices())?;
    let testing = select(dataset, fold.test_indices())?;
    let model = estimator.fit(&training, ())?;
    let predictions = model.predict(testing.records())?;
    let score = scorer(testing.targets(), predictions.view())?;
    if !score.is_finite() {
        return Err(MlError::NonFiniteCrossValidationScore { fold_index });
    }
    Ok(score)
}

pub(super) fn select<Target: Clone>(
    dataset: &Dataset<Target>,
    indices: &[usize],
) -> Result<Dataset<Target>> {
    let records = dataset.records().select(Axis(0), indices);
    let targets = dataset.targets().select(Axis(0), indices);
    Dataset::new(records, targets)
}
