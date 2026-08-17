//! Rank hyperparameter configurations using cross-validation.

use machlearn::{
    Dataset, Fit, KFold, MlError, ParameterGrid, ParameterSet, Predict, Result, ScoreDirection,
    grid_search, mean_absolute_error,
};
use ndarray::{Array1, ArrayView2, array};

struct ConstantRegressor {
    prediction: f64,
}

struct FittedConstantRegressor {
    prediction: f64,
}

impl Fit<&Dataset<f64>, ()> for ConstantRegressor {
    type Fitted = FittedConstantRegressor;

    fn fit(&self, _dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
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

fn configured_estimator(parameters: &ParameterSet) -> Result<ConstantRegressor> {
    let prediction = parameters
        .get("prediction")
        .and_then(machlearn::ParameterValue::as_f64)
        .ok_or(MlError::InvalidParameterName)?;
    Ok(ConstantRegressor { prediction })
}

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0]],
        array![2.0, 2.0, 2.0, 2.0],
    )?;
    let folds = KFold::new(2)?.split(dataset.n_samples())?;
    let grid = ParameterGrid::new().with_parameter("prediction", [0.0, 1.0, 2.0, 3.0])?;
    let result = grid_search(
        &grid,
        configured_estimator,
        &dataset,
        &folds,
        mean_absolute_error,
        ScoreDirection::Minimize,
    )?;

    for entry in result.entries() {
        println!(
            "rank {}: {:?}, mean MAE {:.3}",
            entry.rank(),
            entry.parameters(),
            entry.mean_score()
        );
    }
    Ok(())
}
