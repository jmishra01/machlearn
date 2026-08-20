//! Fitting multinomial Naive Bayes over word-count-like features.

use machlearn::{Dataset, MultinomialNaiveBayes, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 3.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [1.0, 3.0, 1.0],
        ],
        array!["a", "a", "b", "a", "b", "b"],
    )?;
    let model = MultinomialNaiveBayes::new().fit(&dataset)?;
    let records = array![[1.0, 1.0, 1.0], [0.0, 0.0, 5.0]];

    println!("classes: {:?}", model.classes());
    println!("class priors: {:?}", model.class_priors());
    println!("feature log-probabilities: {:?}", model.feature_log_prob());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    Ok(())
}
