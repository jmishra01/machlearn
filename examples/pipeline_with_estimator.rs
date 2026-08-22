//! Chaining preprocessing steps with a final estimator, then predicting.

use machlearn::{Dataset, LinearRegression, Pipeline, Result, StandardScaler};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [1.0, 5.0],
        [2.0, 3.0],
        [3.0, 8.0],
        [4.0, 1.0],
        [5.0, 6.0],
        [6.0, 2.0],
    ];
    let targets = array![17.0, 13.0, 30.0, 11.0, 28.0, 18.0];
    let training = Dataset::new(records, targets)?;

    let pipeline = Pipeline::new()
        .then(StandardScaler::default())
        .with_estimator(LinearRegression::new());
    let fitted = pipeline.fit(&training)?;

    let held_out = array![[7.0, 4.0], [0.5, 9.0]];
    let predictions = fitted.predict(held_out.view())?;

    println!("predictions for held-out rows:\n{predictions:?}");
    Ok(())
}
