//! Growing a bagged forest and inspecting averaged leaf-probability predictions.

use machlearn::{Dataset, MaxFeatures, RandomForestClassifier, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )?;
    let model = RandomForestClassifier::new()
        .with_n_estimators(25)?
        .with_max_features(MaxFeatures::All)?
        .with_seed(7)
        .fit(&dataset)?;
    let records = array![[-0.5], [0.5], [2.5]];

    println!("classes: {:?}", model.classes());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    println!("n_estimators: {}", model.n_estimators());
    println!("feature importances: {:?}", model.feature_importances());
    Ok(())
}
