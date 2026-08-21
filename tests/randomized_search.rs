//! Integration tests for randomized hyperparameter search.

use std::cell::Cell;

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, Fit, KFold, MlError, ParameterGrid, ParameterSet, Predict, Result, ScoreDirection,
    mean_absolute_error, randomized_search,
};
use ndarray::{Array1, ArrayView2, array};

struct ConstantRegressor<'a> {
    prediction: f64,
    fit_count: &'a Cell<usize>,
}

struct FittedConstantRegressor {
    prediction: f64,
}

impl Fit<&Dataset<f64>, ()> for ConstantRegressor<'_> {
    type Fitted = FittedConstantRegressor;

    fn fit(&self, _dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        self.fit_count.set(self.fit_count.get() + 1);
        Ok(FittedConstantRegressor {
            prediction: self.prediction,
        })
    }
}

impl<'a> Predict<ArrayView2<'a, f64>> for FittedConstantRegressor {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Ok(Array1::from_elem(features.nrows(), self.prediction))
    }
}

fn dataset_and_folds() -> (Dataset<f64>, Vec<machlearn::Fold>) {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0]],
        array![2.0, 2.0, 2.0, 2.0],
    )
    .unwrap();
    let folds = KFold::new(2).unwrap().split(dataset.n_samples()).unwrap();
    (dataset, folds)
}

fn prediction(parameters: &ParameterSet) -> f64 {
    parameters.get("prediction").unwrap().as_f64().unwrap()
}

#[test]
fn evaluates_exactly_n_iter_drawn_assignments() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [0.0, 2.0, 4.0])
        .unwrap();
    let fit_count = Cell::new(0);

    let result = randomized_search(
        &grid,
        5,
        7,
        |parameters| {
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
    .unwrap();

    assert_eq!(result.len(), 5);
    assert_eq!(fit_count.get(), 10);
    for entry in result.entries() {
        let value = prediction(entry.parameters());
        assert!([0.0, 2.0, 4.0].contains(&value));
    }
}

#[test]
fn ranks_the_best_scoring_draw_first() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [0.0, 2.0, 4.0])
        .unwrap();
    let fit_count = Cell::new(0);

    let result = randomized_search(
        &grid,
        20,
        1,
        |parameters| {
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
    .unwrap();

    assert_abs_diff_eq!(prediction(result.best().unwrap().parameters()), 2.0);
    assert_abs_diff_eq!(result.best().unwrap().mean_score(), 0.0);
    assert_eq!(result.best().unwrap().rank(), 1);
}

#[test]
fn is_deterministic_for_a_fixed_seed() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [0.0, 1.0, 2.0, 3.0, 4.0])
        .unwrap();
    let fit_count = Cell::new(0);

    let search = |seed: u64| {
        randomized_search(
            &grid,
            6,
            seed,
            |parameters| {
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
        .unwrap()
    };

    let first = search(42);
    let second = search(42);
    assert_eq!(first, second);
}

#[test]
fn evaluates_an_empty_grid_as_n_iter_default_configurations() {
    let (dataset, folds) = dataset_and_folds();
    let fit_count = Cell::new(0);

    let result = randomized_search(
        &ParameterGrid::new(),
        3,
        0,
        |parameters| {
            assert!(parameters.is_empty());
            Ok(ConstantRegressor {
                prediction: 2.0,
                fit_count: &fit_count,
            })
        },
        &dataset,
        &folds,
        mean_absolute_error,
        ScoreDirection::Minimize,
    )
    .unwrap();

    assert_eq!(result.len(), 3);
    assert!(result.entries().iter().all(|entry| entry.rank() == 1));
}

#[test]
fn rejects_a_zero_iteration_count() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [1.0, 2.0])
        .unwrap();
    let fit_count = Cell::new(0);

    let error = randomized_search(
        &grid,
        0,
        0,
        |parameters| {
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
    .unwrap_err();

    assert_eq!(error, MlError::InvalidSearchIterations(0));
}

#[test]
fn propagates_factory_errors_without_evaluating_later_candidates() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [1.0])
        .unwrap();
    let factory_count = Cell::new(0);
    let fit_count = Cell::new(0);

    let error = randomized_search(
        &grid,
        3,
        0,
        |_parameters| {
            factory_count.set(factory_count.get() + 1);
            if factory_count.get() == 2 {
                return Err(MlError::InvalidParameterName);
            }
            Ok(ConstantRegressor {
                prediction: 1.0,
                fit_count: &fit_count,
            })
        },
        &dataset,
        &folds,
        mean_absolute_error,
        ScoreDirection::Minimize,
    )
    .unwrap_err();

    assert_eq!(error, MlError::InvalidParameterName);
    assert_eq!(factory_count.get(), 2);
}

#[cfg(feature = "serde")]
#[test]
fn results_round_trip_through_serde() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [1.0, 2.0])
        .unwrap();
    let fit_count = Cell::new(0);
    let result = randomized_search(
        &grid,
        4,
        3,
        |parameters| {
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
    .unwrap();

    let json = serde_json::to_string(&result).unwrap();
    let restored: machlearn::GridSearchResult = serde_json::from_str(&json).unwrap();

    assert_eq!(result, restored);
}
