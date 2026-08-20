//! Fitting Lasso regression and inspecting its sparse coefficients.

use machlearn::{Dataset, LassoRegression, Result};
use ndarray::array;

fn main() -> Result<()> {
    // The third column is uncorrelated with the target; Lasso is expected to
    // shrink its coefficient to exactly zero.
    let dataset = Dataset::new(
        array![
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 2.0],
            [5.0, 0.0, 0.0],
            [6.0, 1.0, 4.0],
        ],
        array![3.1, 4.8, 7.05, 9.0, 10.9, 13.15],
    )?;
    let model = LassoRegression::new(0.5)?.fit(&dataset)?;

    println!("coefficients: {:?}", model.coefficients());
    println!("intercept: {:.3}", model.intercept());
    println!("nonzero coefficients: {}", model.n_nonzero_coefficients());
    println!(
        "converged in {} iterations",
        model.convergence().iterations()
    );
    println!(
        "predictions: {:?}",
        model.predict(array![[2.5, 0.5, 1.0]].view())?
    );
    Ok(())
}
