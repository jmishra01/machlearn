//! Constructing and inspecting a validated dataset.

use machlearn::{Dataset, Result};
use ndarray::array;

fn main() -> Result<()> {
    let flowers = Dataset::new(
        array![[5.1, 3.5], [7.0, 3.2], [6.3, 3.3]],
        array!["setosa", "versicolor", "virginica"],
    )?;

    println!("shape: {:?}", flowers.shape());
    println!("samples: {}", flowers.n_samples());
    println!("features per sample: {}", flowers.n_features());
    println!("targets: {:?}", flowers.targets());

    Ok(())
}
