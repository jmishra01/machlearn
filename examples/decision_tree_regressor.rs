//! Growing a decision tree and comparing depth-limited predictions.

use machlearn::{Dataset, DecisionTreeRegressor, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
    )?;
    let records = array![[0.5], [3.5]];

    let unrestricted = DecisionTreeRegressor::new().fit(&dataset)?;
    let shallow = DecisionTreeRegressor::new()
        .with_max_depth(Some(0))
        .fit(&dataset)?;

    println!(
        "unrestricted predictions: {:?}",
        unrestricted.predict(records.view())?
    );
    println!(
        "max_depth(0) predictions: {:?}",
        shallow.predict(records.view())?
    );
    println!(
        "feature importances: {:?}",
        unrestricted.feature_importances()
    );
    Ok(())
}
