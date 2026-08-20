//! Fitting linear discriminant analysis and inspecting its linear discriminant scores.

use machlearn::{Dataset, LinearDiscriminantAnalysis, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![
            [-3.0, 0.0],
            [-2.0, 0.5],
            [-2.0, -0.5],
            [3.0, 0.0],
            [2.0, 0.5],
            [2.0, -0.5],
            [0.0, 3.0],
            [0.5, 2.0],
            [-0.5, 2.0],
        ],
        array![
            "amber", "amber", "amber", "blue", "blue", "blue", "cyan", "cyan", "cyan",
        ],
    )?;
    let model = LinearDiscriminantAnalysis::new().fit(&dataset)?;
    let records = array![[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]];

    println!("classes: {:?}", model.classes());
    println!("coefficients: {:?}", model.coefficients());
    println!("intercepts: {:?}", model.intercepts());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    Ok(())
}
