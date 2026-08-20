//! Growing a bagged forest of regression trees and averaging predictions.

use machlearn::{Dataset, RandomForestRegressor, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
    )?;
    let model = RandomForestRegressor::new()
        .with_n_estimators(25)?
        .with_seed(7)
        .fit(&dataset)?;
    let records = array![[0.5], [4.5]];

    println!("predictions: {:?}", model.predict(records.view())?);
    println!("n_estimators: {}", model.n_estimators());
    println!("feature importances: {:?}", model.feature_importances());
    Ok(())
}
