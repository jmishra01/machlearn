//! Fitting and applying an ordered preprocessing pipeline.

use machlearn::{MinMaxScaler, Pipeline, Result, StandardScaler};
use ndarray::array;

fn main() -> Result<()> {
    let training = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    let test = array![[2.0, 15.0], [6.0, 35.0]];

    let pipeline = Pipeline::new()
        .then(StandardScaler::default())
        .then(MinMaxScaler::new(-1.0, 1.0)?);
    // Fit only on training data. Reusing this fitted pipeline for test data
    // prevents test statistics from leaking into preprocessing parameters.
    let fitted = pipeline.fit(training.view())?;
    let transformed_test = fitted.transform(test.view())?;

    println!("pipeline steps: {}", fitted.len());
    println!("transformed test data:\n{transformed_test:?}");
    Ok(())
}
