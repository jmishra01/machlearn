//! Rank randomly drawn parameter configurations with optional Rayon execution.

#[cfg(feature = "parallel")]
fn main() -> machlearn::Result<()> {
    use machlearn::{
        Dataset, Fit, KFold, MlError, ParameterGrid, ParameterSet, Predict, ScoreDirection,
        mean_absolute_error, randomized_search_parallel,
    };
    use ndarray::{Array1, ArrayView2, array};

    struct ConstantRegressor(f64);
    struct FittedConstantRegressor(f64);

    impl Fit<&Dataset<f64>, ()> for ConstantRegressor {
        type Fitted = FittedConstantRegressor;

        fn fit(&self, _dataset: &Dataset<f64>, (): ()) -> machlearn::Result<Self::Fitted> {
            Ok(FittedConstantRegressor(self.0))
        }
    }

    impl<'a> Predict<ArrayView2<'a, f64>> for FittedConstantRegressor {
        type Output = Array1<f64>;

        fn predict(&self, features: ArrayView2<'a, f64>) -> machlearn::Result<Self::Output> {
            Ok(Array1::from_elem(features.nrows(), self.0))
        }
    }

    fn configured(parameters: &ParameterSet) -> machlearn::Result<ConstantRegressor> {
        parameters
            .get("prediction")
            .and_then(machlearn::ParameterValue::as_f64)
            .map(ConstantRegressor)
            .ok_or(MlError::InvalidParameterName)
    }

    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0]],
        array![2.0, 2.0, 2.0, 2.0],
    )?;
    let folds = KFold::new(2)?.split(dataset.n_samples())?;
    let grid = ParameterGrid::new().with_parameter(
        "prediction",
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    )?;
    let result = randomized_search_parallel(
        &grid,
        5,
        42,
        configured,
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

#[cfg(not(feature = "parallel"))]
fn main() {
    println!("Run with: cargo run --example parallel_randomized_search --features parallel");
}
