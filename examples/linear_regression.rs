//! Fit and use an ordinary least-squares regression model.

use machlearn::{Dataset, LinearRegression, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0]],
        array![1.0, 3.0, 5.0, 7.0, 9.0],
    )?;
    let model = LinearRegression::new().fit(&dataset)?;
    let predictions = model.predict(array![[5.0], [6.0]].view())?;

    println!("intercept: {:.3}", model.intercept());
    println!("coefficients: {:?}", model.coefficients());
    println!("predictions: {predictions:?}");
    Ok(())
}
