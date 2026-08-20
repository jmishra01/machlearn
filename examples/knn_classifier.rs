//! Voting among nearby training points and comparing weighting schemes.

use machlearn::{Dataset, KNeighborsClassifier, Result, Weighting};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )?;
    let records = array![[-0.5], [0.5], [2.5]];

    let uniform = KNeighborsClassifier::new(3)?.fit(&dataset)?;
    let distance = KNeighborsClassifier::new(3)?
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
    println!("training samples retained: {}", uniform.n_samples());
    Ok(())
}
