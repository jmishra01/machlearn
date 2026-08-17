//! Fit and score a regression estimator independently in each fold.

use machlearn::{Dataset, Fit, KFold, Predict, Result, cross_validate, mean_absolute_error};
use ndarray::{Array1, ArrayView2, array};

struct MeanRegressor;

struct FittedMeanRegressor {
    mean: f64,
}

impl Fit<&Dataset<f64>, ()> for MeanRegressor {
    type Fitted = FittedMeanRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
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

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
        array![1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    )?;
    let folds = KFold::new(3)?
        .with_shuffle(true)
        .with_seed(7)
        .split(dataset.n_samples())?;
    let scores = cross_validate(&MeanRegressor, &dataset, &folds, mean_absolute_error)?;

    println!("fold MAE: {:?}", scores.scores());
    println!("mean MAE: {:.4}", scores.mean());
    println!("standard deviation: {:.4}", scores.standard_deviation());
    Ok(())
}
