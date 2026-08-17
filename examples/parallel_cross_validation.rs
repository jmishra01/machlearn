//! Score independent folds with the optional Rayon execution feature.

#[cfg(feature = "parallel")]
fn main() -> machlearn::Result<()> {
    use machlearn::{Dataset, Fit, KFold, Predict, cross_validate_parallel, mean_absolute_error};
    use ndarray::{Array1, ArrayView2, array};

    struct MeanRegressor;
    struct FittedMeanRegressor(f64);

    impl Fit<&Dataset<f64>, ()> for MeanRegressor {
        type Fitted = FittedMeanRegressor;

        fn fit(&self, dataset: &Dataset<f64>, (): ()) -> machlearn::Result<Self::Fitted> {
            #[allow(clippy::cast_precision_loss)]
            let count = dataset.n_samples() as f64;
            Ok(FittedMeanRegressor(
                dataset.targets().iter().sum::<f64>() / count,
            ))
        }
    }

    impl<'a> Predict<ArrayView2<'a, f64>> for FittedMeanRegressor {
        type Output = Array1<f64>;

        fn predict(&self, features: ArrayView2<'a, f64>) -> machlearn::Result<Self::Output> {
            Ok(Array1::from_elem(features.nrows(), self.0))
        }
    }

    let dataset = Dataset::new(
        array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
        array![1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    )?;
    let folds = KFold::new(3)?.split(dataset.n_samples())?;
    let scores = cross_validate_parallel(&MeanRegressor, &dataset, &folds, mean_absolute_error)?;

    println!("parallel fold MAE: {:?}", scores.scores());
    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn main() {
    println!("Run with: cargo run --example parallel_cross_validation --features parallel");
}
