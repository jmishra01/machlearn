use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{
    CurveScores, Fold,
    cross_validation::{select, validate_folds},
};
use machlearn_core::core::{Dataset, Fit, MlError, Predict, Result};

/// Scores a family of estimators built by sweeping one hyperparameter,
/// fitted and evaluated on every fold, reporting both train and test
/// scores.
///
/// `estimator_factory` builds a configured estimator from each entry of
/// `param_range` in turn; every resulting estimator is fitted independently
/// on each fold's training rows (the same isolation guarantees as
/// [`super::cross_validate`]) and scored on both that training subset and
/// the fold's held-out test rows. Comparing the two across `param_range` is
/// the classic bias/variance diagnostic: a hyperparameter value where train
/// and test scores diverge is overfitting; where both are poor, the model
/// is underfitting.
///
/// # Errors
///
/// Returns an error when `param_range` is empty, folds are invalid,
/// estimator construction, fitting, or prediction fails, or the scorer
/// fails or returns a non-finite value.
pub fn validation_curve<Target, Value, Factory, Estimator, Model, Prediction, Scorer>(
    param_range: &[Value],
    estimator_factory: Factory,
    dataset: &Dataset<Target>,
    folds: &[Fold],
    scorer: Scorer,
) -> Result<CurveScores>
where
    Target: Clone,
    Factory: Fn(&Value) -> Result<Estimator>,
    for<'dataset> Estimator: Fit<&'dataset Dataset<Target>, (), Fitted = Model>,
    for<'features> Model: Predict<ArrayView2<'features, f64>, Output = Array1<Prediction>>,
    for<'actual, 'predicted> Scorer:
        Fn(ArrayView1<'actual, Target>, ArrayView1<'predicted, Prediction>) -> Result<f64>,
{
    validate_folds(folds, dataset.n_samples())?;
    if param_range.is_empty() {
        return Err(MlError::EmptyCurvePoints);
    }

    let mut train_scores = Array2::zeros((param_range.len(), folds.len()));
    let mut test_scores = Array2::zeros((param_range.len(), folds.len()));
    for (point_index, value) in param_range.iter().enumerate() {
        let estimator = estimator_factory(value)?;
        for (fold_index, fold) in folds.iter().enumerate() {
            let training = select(dataset, fold.train_indices())?;
            let testing = select(dataset, fold.test_indices())?;
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
