//! Integration tests for optional parallel model selection.

#![cfg(feature = "parallel")]

use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};

use machlearn::{
    Dataset, Fit, KFold, MlError, ParameterGrid, ParameterSet, Predict, Result, ScoreDirection,
    cross_validate, cross_validate_parallel, grid_search, grid_search_parallel,
    mean_absolute_error,
};
use ndarray::{Array1, ArrayView2, array};

struct MeanRegressor<'a> {
    fit_count: &'a AtomicUsize,
}

struct ConstantRegressor<'a> {
    prediction: f64,
    fit_count: &'a AtomicUsize,
}

struct BarrierRegressor {
    barrier: Arc<Barrier>,
}

struct FittedConstantRegressor {
    prediction: f64,
}

impl Fit<&Dataset<f64>, ()> for MeanRegressor<'_> {
    type Fitted = FittedConstantRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        self.fit_count.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let count = dataset.n_samples() as f64;
        Ok(FittedConstantRegressor {
            prediction: dataset.targets().iter().sum::<f64>() / count,
        })
    }
}

impl Fit<&Dataset<f64>, ()> for ConstantRegressor<'_> {
    type Fitted = FittedConstantRegressor;

    fn fit(&self, _dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        self.fit_count.fetch_add(1, Ordering::Relaxed);
        Ok(FittedConstantRegressor {
            prediction: self.prediction,
        })
    }
}

impl Fit<&Dataset<f64>, ()> for BarrierRegressor {
    type Fitted = FittedConstantRegressor;

    fn fit(&self, _dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        self.barrier.wait();
        Ok(FittedConstantRegressor { prediction: 0.0 })
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedConstantRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Ok(Array1::from_elem(features.nrows(), self.prediction))
    }
}

fn dataset() -> Dataset<f64> {
    Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]],
        array![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
    )
    .unwrap()
}

fn prediction(parameters: &ParameterSet) -> f64 {
    parameters.get("prediction").unwrap().as_f64().unwrap()
}

#[test]
fn parallel_cross_validation_matches_sequential_results() {
    let dataset = dataset();
    let folds = KFold::new(4).unwrap().split(dataset.n_samples()).unwrap();
    let sequential_fits = AtomicUsize::new(0);
    let parallel_fits = AtomicUsize::new(0);

    let sequential = cross_validate(
        &MeanRegressor {
            fit_count: &sequential_fits,
        },
        &dataset,
        &folds,
        mean_absolute_error,
    )
    .unwrap();
    let parallel = cross_validate_parallel(
        &MeanRegressor {
            fit_count: &parallel_fits,
        },
        &dataset,
        &folds,
        mean_absolute_error,
    )
    .unwrap();

    assert_eq!(parallel, sequential);
    assert_eq!(sequential_fits.load(Ordering::Relaxed), 4);
    assert_eq!(parallel_fits.load(Ordering::Relaxed), 4);
}

#[test]
fn parallel_cross_validation_preserves_error_order() {
    let dataset = dataset();
    let folds = KFold::new(4).unwrap().split(dataset.n_samples()).unwrap();
    let fit_count = AtomicUsize::new(0);

    let error = cross_validate_parallel(
        &MeanRegressor {
            fit_count: &fit_count,
        },
        &dataset,
        &folds,
        |_actual, _predicted| Ok(f64::NAN),
    )
    .unwrap_err();

    assert_eq!(
        error,
        MlError::NonFiniteCrossValidationScore { fold_index: 0 }
    );
}

#[test]
fn parallel_grid_search_matches_sequential_rankings() {
    let dataset = dataset();
    let folds = KFold::new(4).unwrap().split(dataset.n_samples()).unwrap();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [30.0, 10.0, 20.0, 0.0])
        .unwrap();
    let sequential_fits = AtomicUsize::new(0);
    let parallel_fits = AtomicUsize::new(0);

    let sequential = grid_search(
        &grid,
        |parameters| {
            Ok(ConstantRegressor {
                prediction: prediction(parameters),
                fit_count: &sequential_fits,
            })
        },
        &dataset,
        &folds,
        mean_absolute_error,
        ScoreDirection::Minimize,
    )
    .unwrap();
    let parallel = grid_search_parallel(
        &grid,
        |parameters| {
            Ok(ConstantRegressor {
                prediction: prediction(parameters),
                fit_count: &parallel_fits,
            })
        },
        &dataset,
        &folds,
        mean_absolute_error,
        ScoreDirection::Minimize,
    )
    .unwrap();

    assert_eq!(parallel, sequential);
    assert_eq!(sequential_fits.load(Ordering::Relaxed), 16);
    assert_eq!(parallel_fits.load(Ordering::Relaxed), 16);
}

#[test]
fn fold_evaluation_executes_concurrently() {
    let dataset = dataset();
    let folds = KFold::new(4).unwrap().split(dataset.n_samples()).unwrap();
    let estimator = BarrierRegressor {
        barrier: Arc::new(Barrier::new(4)),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let result =
        pool.install(|| cross_validate_parallel(&estimator, &dataset, &folds, mean_absolute_error));

    assert!(result.is_ok());
}

#[test]
fn parameter_factories_execute_concurrently() {
    let dataset = dataset();
    let folds = KFold::new(2).unwrap().split(dataset.n_samples()).unwrap();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [0.0, 1.0, 2.0, 3.0])
        .unwrap();
    let barrier = Arc::new(Barrier::new(4));
    let fit_count = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let result = pool.install(|| {
        grid_search_parallel(
            &grid,
            |parameters| {
                barrier.wait();
                Ok(ConstantRegressor {
                    prediction: prediction(parameters),
                    fit_count: &fit_count,
                })
            },
            &dataset,
            &folds,
            mean_absolute_error,
            ScoreDirection::Minimize,
        )
    });

    assert!(result.is_ok());
    assert_eq!(fit_count.load(Ordering::Relaxed), 8);
}
