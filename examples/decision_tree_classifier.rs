//! Growing a decision tree and inspecting leaf-probability predictions and
//! feature importances.

use machlearn::{Dataset, DecisionTreeClassifier, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )?;
    let model = DecisionTreeClassifier::new()
        .with_max_depth(Some(3))
        .fit(&dataset)?;
    let records = array![[-0.5], [0.5], [2.5]];

    println!("classes: {:?}", model.classes());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    println!("feature importances: {:?}", model.feature_importances());
    Ok(())
}
