//! Integration tests for cross-validation scoring.

use std::cell::Cell;

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, Fit, KFold, MlError, Predict, Result, StratifiedKFold, cross_validate,
    mean_absolute_error,
};
use ndarray::{Array1, ArrayView2, array};

struct CountingMeanRegressor<'a> {
    fit_count: &'a Cell<usize>,
}

struct FittedMeanRegressor {
    mean: f64,
}

impl Fit<&Dataset<f64>, ()> for CountingMeanRegressor<'_> {
    type Fitted = FittedMeanRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        self.fit_count.set(self.fit_count.get() + 1);
        #[allow(clippy::cast_precision_loss)]
        let count = dataset.n_samples() as f64;
        Ok(FittedMeanRegressor {
            mean: dataset.targets().iter().sum::<f64>() / count,
        })
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedMeanRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Ok(Array1::from_elem(features.nrows(), self.mean))
    }
}

fn regression_dataset(targets: Array1<f64>) -> Dataset<f64> {
    #[allow(clippy::cast_precision_loss)]
    let records = Array1::from_iter((0..targets.len()).map(|value| value as f64))
        .insert_axis(ndarray::Axis(1));
    Dataset::new(records, targets).unwrap()
}

#[test]
fn fits_a_fresh_model_for_every_fold() {
    let dataset = regression_dataset(array![0.0, 0.0, 10.0, 10.0]);
    let folds = KFold::new(2).unwrap().split(dataset.n_samples()).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };

    let result = cross_validate(&estimator, &dataset, &folds, mean_absolute_error).unwrap();

    assert_eq!(fit_count.get(), 2);
    assert_eq!(result.scores(), &[10.0, 10.0]);
    assert_abs_diff_eq!(result.mean(), 10.0);
    assert_abs_diff_eq!(result.standard_deviation(), 0.0);
}

#[test]
fn reports_scores_in_fold_order() {
    let dataset = regression_dataset(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let folds = KFold::new(3).unwrap().split(dataset.n_samples()).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };

    let result = cross_validate(&estimator, &dataset, &folds, |actual, _predicted| {
        Ok(actual[0])
    })
    .unwrap();

    assert_eq!(result.scores(), &[1.0, 3.0, 5.0]);
    assert_abs_diff_eq!(result.mean(), 3.0);
    assert_abs_diff_eq!(result.standard_deviation(), (8.0_f64 / 3.0).sqrt());
}

#[test]
fn accepts_stratified_folds() {
    let dataset = regression_dataset(array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let labels = [0, 0, 0, 1, 1, 1];
    let folds = StratifiedKFold::new(3).unwrap().split(&labels).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };

    let result = cross_validate(&estimator, &dataset, &folds, mean_absolute_error).unwrap();

    assert_eq!(result.scores().len(), 3);
    assert_eq!(fit_count.get(), 3);
}

#[test]
fn rejects_fewer_than_two_folds_before_fitting() {
    let dataset = regression_dataset(array![1.0, 2.0, 3.0, 4.0]);
    let folds = KFold::new(2).unwrap().split(dataset.n_samples()).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };

    let error = cross_validate(&estimator, &dataset, &folds[..1], mean_absolute_error).unwrap_err();

    assert_eq!(error, MlError::InvalidFoldCount { n_splits: 1 });
    assert_eq!(fit_count.get(), 0);
}

#[test]
fn rejects_folds_built_for_a_different_dataset_size() {
    let dataset = regression_dataset(array![1.0, 2.0, 3.0, 4.0, 5.0]);
    let folds = KFold::new(2).unwrap().split(4).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };

    let error = cross_validate(&estimator, &dataset, &folds, mean_absolute_error).unwrap_err();

    assert_eq!(error, MlError::InvalidFoldPartition { fold_index: 0 });
    assert_eq!(fit_count.get(), 0);
}

#[test]
fn rejects_non_finite_scores() {
    let dataset = regression_dataset(array![1.0, 2.0, 3.0, 4.0]);
    let folds = KFold::new(2).unwrap().split(dataset.n_samples()).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };

    let error = cross_validate(&estimator, &dataset, &folds, |_actual, _predicted| {
        Ok(f64::NAN)
    })
    .unwrap_err();

    assert_eq!(
        error,
        MlError::NonFiniteCrossValidationScore { fold_index: 0 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn scores_round_trip_through_serde() {
    let dataset = regression_dataset(array![1.0, 2.0, 3.0, 4.0]);
    let folds = KFold::new(2).unwrap().split(dataset.n_samples()).unwrap();
    let fit_count = Cell::new(0);
    let estimator = CountingMeanRegressor {
        fit_count: &fit_count,
    };
    let result = cross_validate(&estimator, &dataset, &folds, mean_absolute_error).unwrap();

    let json = serde_json::to_string(&result).unwrap();
    let restored: machlearn::CrossValidationScores = serde_json::from_str(&json).unwrap();

    assert_eq!(result, restored);
}
