//! Checking whether a regularization strength underfits or overfits.

use machlearn::{
    Dataset, KFold, ParameterValue, Result, RidgeRegression, mean_squared_error, validation_curve,
};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ],
        array![1.0, 2.9, 5.1, 7.0, 9.2, 11.0, 12.8, 15.1, 17.0, 19.2],
    )?;
    let folds = KFold::new(3)?.split(dataset.n_samples())?;
    let alphas = [
        ParameterValue::from(0.01),
        ParameterValue::from(1.0),
        ParameterValue::from(100.0),
    ];

    let scores = validation_curve(
        &alphas,
        |value| RidgeRegression::new(value.as_f64().unwrap()),
        &dataset,
        &folds,
        mean_squared_error,
    )?;

    println!("train_scores_mean: {:?}", scores.train_scores_mean());
    println!("test_scores_mean:  {:?}", scores.test_scores_mean());
    Ok(())
}
