//! Expand named hyperparameter candidates in deterministic order.

use machlearn::{ParameterGrid, Result};

fn main() -> Result<()> {
    let grid = ParameterGrid::new()
        .with_parameter("fit_intercept", [true, false])?
        .with_parameter("regularization", [0.0, 0.01, 0.1])?;

    println!("{} configurations", grid.combination_count()?);
    for (index, parameters) in grid.combinations()?.iter().enumerate() {
        println!("{}: {:?}", index + 1, parameters);
    }
    Ok(())
}
