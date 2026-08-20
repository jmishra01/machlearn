//! Sequentially boosting shallow regression trees to fit a nonlinear target.

use machlearn::{Dataset, GradientBoostingRegressor, Result};
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
        array![1.0, 1.2, 1.5, 4.8, 5.1, 5.3, 9.9, 10.1, 10.4, 10.0],
    )?;
    let model = GradientBoostingRegressor::new()
        .with_n_estimators(20)?
        .with_learning_rate(0.1)?
        .fit(&dataset)?;
    let records = array![[0.5], [3.5], [8.5]];

    println!("predictions: {:?}", model.predict(records.view())?);
    println!("n_estimators: {}", model.n_estimators());
    println!("initial prediction: {}", model.initial_prediction());
    Ok(())
}
