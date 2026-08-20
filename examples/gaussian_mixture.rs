//! Fitting a Gaussian mixture model and inspecting soft cluster assignments.

use machlearn::{GaussianMixture, Result};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [-3.0, -3.0],
        [-2.5, -2.5],
        [-3.5, -2.0],
        [-2.8, -3.2],
        [-3.2, -2.8],
        [3.0, 3.0],
        [2.5, 2.5],
        [3.5, 2.0],
        [2.8, 3.2],
        [3.2, 2.8],
    ];
    let model = GaussianMixture::new(2)?.fit(records.view())?;

    println!("weights: {:?}", model.weights());
    println!("means: {:?}", model.means());
    println!(
        "iterations: {} (converged: {})",
        model.n_iterations(),
        model.converged()
    );
    println!("log_likelihood: {:.4}", model.log_likelihood());

    let query = array![[-3.0, -2.8], [3.1, 2.9]];
    println!(
        "membership probabilities: {:?}",
        model.predict_probabilities(query.view())?
    );
    println!("predictions: {:?}", model.predict(query.view())?);
    Ok(())
}
