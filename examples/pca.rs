//! Reducing dimensionality and inspecting explained-variance reporting.

use machlearn::{PrincipalComponentAnalysis, Result};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ];
    let model = PrincipalComponentAnalysis::new()
        .with_n_components(Some(1))?
        .fit(records.view())?;

    println!("components: {:?}", model.components());
    println!("explained variance: {:?}", model.explained_variance());
    println!(
        "explained variance ratio: {:?}",
        model.explained_variance_ratio()
    );

    let projected = model.transform(records.view())?;
    println!("projected: {projected:?}");
    println!(
        "reconstructed: {:?}",
        model.inverse_transform(projected.view())?
    );
    Ok(())
}
