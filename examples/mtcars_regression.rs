//! A worked regression example on the 1974 Motor Trend US road-test data
//! (a real, public-domain dataset bundled at `examples/data/mtcars.csv`),
//! not synthetic data.

#[cfg(feature = "csv")]
fn main() -> machlearn::Result<()> {
    use machlearn::{
        Dataset, RidgeRegression, SplitOptions, StandardScaler, dataset_from_csv_path,
        mean_absolute_error, r2_score, train_test_split,
    };

    // Ten real car specs (cylinders, displacement, horsepower, weight, and
    // so on) predicting fuel economy (miles per US gallon).
    let dataset: Dataset<f64> = dataset_from_csv_path("examples/data/mtcars.csv", true, 0)?;
    println!(
        "loaded {} samples, {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    let (train, test) = train_test_split(&dataset, SplitOptions::new(0.25).with_seed(0))?;

    // Fit the scaler on the training split only, then apply it to both
    // splits, so no information about the test split leaks into training.
    let scaler = StandardScaler::default().fit(train.records())?;
    let scaled_train = Dataset::new(
        scaler.transform(train.records())?,
        train.targets().to_owned(),
    )?;
    let scaled_test = scaler.transform(test.records())?;

    let model = RidgeRegression::new(1.0)?.fit(&scaled_train)?;
    let predictions = model.predict(scaled_test.view())?;

    let r_squared = r2_score(test.targets(), predictions.view())?;
    let mae = mean_absolute_error(test.targets(), predictions.view())?;
    println!("test R-squared: {r_squared:.4}");
    println!("test mean absolute error: {mae:.4} mpg");

    println!(
        "coefficients (cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb): {:?}",
        model.coefficients()
    );
    Ok(())
}

#[cfg(not(feature = "csv"))]
fn main() {
    println!("Run with: cargo run --example mtcars_regression --features csv");
}
