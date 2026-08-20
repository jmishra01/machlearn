//! Fitting a binary classifier and inspecting class probabilities.

use machlearn::{Dataset, LogisticRegression, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )?;
    let estimator = LogisticRegression::new()
        .with_max_iterations(50)?
        .with_tolerance(1.0e-8)?;
    let model = estimator.fit(&dataset)?;
    let records = array![[-0.5], [0.5], [2.5]];

    println!("classes: {:?}", model.classes());
    println!("coefficients: {:?}", model.coefficients());
    println!("intercept: {:.3}", model.intercept());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    println!(
        "converged in {} iterations (tolerance {:.1e})",
        model.convergence().iterations(),
        model.convergence().tolerance()
    );
    Ok(())
}
