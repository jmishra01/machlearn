//! Partitioning well-separated points and inspecting cluster diagnostics.

use machlearn::{KMeans, Result};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [-3.0, -3.0],
        [-2.5, -2.5],
        [-3.5, -2.0],
        [3.0, 3.0],
        [2.5, 2.5],
        [3.5, 2.0],
    ];
    let model = KMeans::new(2)?.fit(records.view())?;

    println!("centroids: {:?}", model.centroids());
    println!("assignments: {:?}", model.predict(records.view())?);
    println!("inertia: {:.4}", model.inertia());
    println!(
        "iterations: {} (converged: {})",
        model.iterations(),
        model.converged()
    );
    Ok(())
}
