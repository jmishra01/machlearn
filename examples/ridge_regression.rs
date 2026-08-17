//! Fit an L2-regularized linear regression model.

use machlearn::{Dataset, Result, RidgeRegression};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0]],
        array![1.0, 3.2, 4.8, 7.1, 8.9],
    )?;
    let model = RidgeRegression::new(0.5)?.fit(&dataset)?;
    let predictions = model.predict(array![[5.0], [6.0]].view())?;

    println!("alpha: {:.3}", model.alpha());
    println!("intercept: {:.3}", model.intercept());
    println!("coefficients: {:?}", model.coefficients());
    println!("predictions: {predictions:?}");
    Ok(())
}
