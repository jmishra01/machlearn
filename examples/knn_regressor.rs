//! Averaging nearby training targets and comparing weighting schemes.

use machlearn::{Dataset, KNeighborsRegressor, Result, Weighting};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0]],
        array![0.0, 1.0, 4.0, 9.0, 16.0],
    )?;
    let records = array![[0.2], [2.2], [3.8]];

    let uniform = KNeighborsRegressor::new(2)?.fit(&dataset)?;
    let distance = KNeighborsRegressor::new(2)?
        .with_weighting(Weighting::Distance)
        .fit(&dataset)?;

    println!(
        "uniform predictions: {:?}",
        uniform.predict(records.view())?
    );
    println!(
        "distance predictions: {:?}",
        distance.predict(records.view())?
    );
    println!("n_neighbors: {}", uniform.n_neighbors());
    Ok(())
}
