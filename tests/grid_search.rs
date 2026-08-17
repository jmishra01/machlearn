//! Integration tests for deterministic hyperparameter grid search.

use std::cell::Cell;

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, Fit, KFold, MlError, ParameterGrid, ParameterSet, Predict, Result, ScoreDirection,
    grid_search, mean_absolute_error,
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
fn minimizes_scores_and_preserves_grid_order_for_ties() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [4.0, 2.0, 0.0])
        .unwrap();
    let factory_count = Cell::new(0);
    let fit_count = Cell::new(0);

    let result = grid_search(
        &grid,
        |parameters| {
            factory_count.set(factory_count.get() + 1);
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

    assert_eq!(factory_count.get(), 3);
    assert_eq!(fit_count.get(), 6);
    assert_eq!(result.len(), 3);
    assert!(!result.is_empty());
    assert_eq!(result.direction(), ScoreDirection::Minimize);
    assert_eq!(
        result
            .entries()
            .iter()
            .map(|entry| prediction(entry.parameters()))
            .collect::<Vec<_>>(),
        vec![2.0, 4.0, 0.0]
    );
    assert_eq!(
        result
            .entries()
            .iter()
            .map(machlearn::GridSearchEntry::rank)
            .collect::<Vec<_>>(),
        vec![1, 2, 2]
    );
    assert_abs_diff_eq!(result.best().unwrap().mean_score(), 0.0);
    assert_eq!(result.best().unwrap().scores().scores(), &[0.0, 0.0]);
}

#[test]
fn supports_maximizing_a_score() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [0.0, 2.0, 4.0])
        .unwrap();
    let fit_count = Cell::new(0);

    let result = grid_search(
        &grid,
        |parameters| {
            Ok(ConstantRegressor {
                prediction: prediction(parameters),
                fit_count: &fit_count,
            })
        },
        &dataset,
        &folds,
        |actual, predicted| Ok(-mean_absolute_error(actual, predicted)?),
        ScoreDirection::Maximize,
    )
    .unwrap();

    assert_abs_diff_eq!(prediction(result.best().unwrap().parameters()), 2.0);
    assert_abs_diff_eq!(result.best().unwrap().mean_score(), 0.0);
}

#[test]
fn evaluates_an_empty_grid_as_one_default_configuration() {
    let (dataset, folds) = dataset_and_folds();
    let fit_count = Cell::new(0);

    let result = grid_search(
        &ParameterGrid::new(),
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

    assert_eq!(result.len(), 1);
    assert_eq!(result.best().unwrap().rank(), 1);
    assert!(result.best().unwrap().parameters().is_empty());
}

#[test]
fn propagates_factory_errors_without_evaluating_later_candidates() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [1.0, 2.0, 3.0])
        .unwrap();
    let factory_count = Cell::new(0);
    let fit_count = Cell::new(0);

    let error = grid_search(
        &grid,
        |parameters| {
            factory_count.set(factory_count.get() + 1);
            let prediction = prediction(parameters);
            if (prediction - 2.0).abs() < f64::EPSILON {
                return Err(MlError::InvalidParameterName);
            }
            Ok(ConstantRegressor {
                prediction,
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
    assert_eq!(fit_count.get(), 2);
}

#[cfg(feature = "serde")]
#[test]
fn results_round_trip_through_serde() {
    let (dataset, folds) = dataset_and_folds();
    let grid = ParameterGrid::new()
        .with_parameter("prediction", [1.0, 2.0])
        .unwrap();
    let fit_count = Cell::new(0);
    let result = grid_search(
        &grid,
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
