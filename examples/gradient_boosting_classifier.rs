//! Sequentially boosting shallow regression trees against log loss to
//! classify two separated groups.

use machlearn::{Dataset, GradientBoostingClassifier, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "no", "yes", "yes", "yes", "yes"],
    )?;
    let model = GradientBoostingClassifier::new()
        .with_n_estimators(20)?
        .with_learning_rate(0.1)?
        .fit(&dataset)?;
    let records = array![[-0.25], [0.25]];

    println!("classes: {:?}", model.classes());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    println!("n_estimators: {}", model.n_estimators());
    Ok(())
}
