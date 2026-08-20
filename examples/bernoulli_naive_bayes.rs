//! Fitting Bernoulli Naive Bayes over binarized presence/absence features.

use machlearn::{BernoulliNaiveBayes, Dataset, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        array!["a", "a", "b", "a", "b", "b"],
    )?;
    let model = BernoulliNaiveBayes::new().fit(&dataset)?;
    let records = array![[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]];

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
