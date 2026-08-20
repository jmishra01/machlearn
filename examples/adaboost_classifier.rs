//! Boosting decision stumps with `AdaBoost`'s discrete SAMME algorithm.

use machlearn::{AdaBoostClassifier, Dataset, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![
            [-3.0],
            [-2.0],
            [-1.0],
            [-0.5],
            [0.4],
            [0.6],
            [1.0],
            [1.5],
            [2.0],
            [3.0]
        ],
        array![
            "no", "no", "no", "yes", "no", "yes", "yes", "no", "yes", "yes"
        ],
    )?;
    let model = AdaBoostClassifier::new()
        .with_n_estimators(6)?
        .with_learning_rate(1.0)?
        .fit(&dataset)?;
    let records = array![[-0.25], [0.25], [1.2]];

    println!("classes: {:?}", model.classes());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    println!("n_estimators used: {}", model.n_estimators());
    Ok(())
}
