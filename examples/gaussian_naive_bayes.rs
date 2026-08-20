//! Fitting Gaussian Naive Bayes and inspecting per-class distributions.

use machlearn::{Dataset, GaussianNaiveBayes, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array!["no", "no", "no", "yes", "yes", "yes"],
    )?;
    let model = GaussianNaiveBayes::new().fit(&dataset)?;
    let records = array![[-0.5], [0.5], [2.5]];

    println!("classes: {:?}", model.classes());
    println!("means: {:?}", model.means());
    println!("variances: {:?}", model.variances());
    println!("class priors: {:?}", model.class_priors());
    println!(
        "probabilities: {:?}",
        model.predict_probabilities(records.view())?
    );
    println!("predictions: {:?}", model.predict(records.view())?);
    Ok(())
}
