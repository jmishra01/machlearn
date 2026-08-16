//! Implementing `MachLearn`'s estimator and fitted-model contracts.

use machlearn::{Dataset, Fit, Predict, Result, SplitOptions, train_test_split};
use ndarray::{Array1, ArrayView2, array};

struct MeanRegressor;

struct FittedMeanRegressor {
    mean: f64,
}

impl Fit<&Dataset<f64>, ()> for MeanRegressor {
    type Fitted = FittedMeanRegressor;

    fn fit(&self, dataset: &Dataset<f64>, (): ()) -> Result<Self::Fitted> {
        #[allow(clippy::cast_precision_loss)]
        let sample_count = dataset.n_samples() as f64;
        let mean = dataset.targets().iter().sum::<f64>() / sample_count;
        Ok(FittedMeanRegressor { mean })
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
        array![[1.0], [2.0], [3.0], [4.0], [5.0]],
        array![2.0, 4.0, 6.0, 8.0, 10.0],
    )?;
    let (train, test) = train_test_split(&dataset, SplitOptions::new(0.4).with_shuffle(false))?;

    let model = MeanRegressor.fit(&train, ())?;
    let predictions = model.predict(test.records())?;

    println!("training target mean: {}", model.mean);
    println!("predictions: {predictions:?}");
    Ok(())
}
