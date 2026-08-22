use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{
    CurveScores, Fold,
    cross_validation::{select, validate_folds},
};
use machlearn_core::core::{Dataset, Fit, MlError, Predict, Result};

/// Scores one estimator fitted on growing training-set sizes, evaluated on
/// every fold, reporting both train and test scores.
///
/// For every entry of `train_sizes` (an absolute row count) and every fold,
/// `estimator` is fitted on the first `train_sizes[i]` rows of that fold's
/// training rows and scored on both that subset and the fold's held-out
/// test rows. Comparing the two across `train_sizes` is the classic
/// diagnostic for whether more training data would help: scores still
/// converging as size grows suggests it would; a persistent gap between
/// train and test scores suggests overfitting that more data alone will
/// not fix.
///
/// # Errors
///
/// Returns an error when `train_sizes` is empty, contains a size larger
/// than some fold's available training rows, folds are invalid, fitting or
/// prediction fails, or the scorer fails or returns a non-finite value.
pub fn learning_curve<Target, Estimator, Model, Prediction, Scorer>(
    estimator: &Estimator,
    train_sizes: &[usize],
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: Scorer,
) -> Result<CurveScores>
where
    Target: Clone,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_folds(folds, dataset.n_samples())?;
    if train_sizes.is_empty() {
        return Err(MlError::EmptyCurvePoints);
    }

    let mut train_scores = Array2::zeros((train_sizes.len(), folds.len()));
    let mut test_scores = Array2::zeros((train_sizes.len(), folds.len()));
    for (fold_index, fold) in folds.iter().enumerate() {
        let testing = select(dataset, fold.test_indices())?;
        for (point_index, &size) in train_sizes.iter().enumerate() {
            if size > fold.train_indices().len() {
                return Err(MlError::InsufficientSamples {
                    required: size,
                    actual: fold.train_indices().len(),
                });
            }
            let training = select(dataset, &fold.train_indices()[..size])?;
            let model = estimator.fit(&training, ())?;
            let train_predictions = model.predict(training.records())?;
            let test_predictions = model.predict(testing.records())?;
            train_scores[[point_index, fold_index]] =
                scorer(training.targets(), train_predictions.view())?;
            test_scores[[point_index, fold_index]] =
                scorer(testing.targets(), test_predictions.view())?;
        }
    }
    Ok(CurveScores::new(train_scores, test_scores))
}
