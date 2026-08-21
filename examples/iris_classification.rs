//! A worked classification example on Fisher's Iris flower measurements
//! (Anderson, 1935; a real, public-domain dataset bundled at
//! `examples/data/iris.csv`), not synthetic data.

#[cfg(feature = "csv")]
fn main() -> machlearn::Result<()> {
    use machlearn::{
        RandomForestClassifier, SplitOptions, accuracy_score, classification_report,
        confusion_matrix, dataset_from_csv_path, train_test_split,
    };

    // Four real petal/sepal measurements (centimeters) predicting one of
    // three iris species.
    let dataset: machlearn::Dataset<String> =
        dataset_from_csv_path("examples/data/iris.csv", true, 4)?;
    println!(
        "loaded {} samples, {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    let (train, test) = train_test_split(&dataset, SplitOptions::new(0.3).with_seed(0))?;

    let model = RandomForestClassifier::new()
        .with_n_estimators(100)?
        .with_seed(0)
        .fit(&train)?;

    let predictions = model.predict(test.records())?;
    let accuracy = accuracy_score(test.targets(), predictions.view())?;
    println!("test accuracy: {accuracy:.4}");
    println!(
        "feature importances (sepal_length, sepal_width, petal_length, petal_width): {:?}",
        model.feature_importances()
    );

    let matrix = confusion_matrix(test.targets(), predictions.view())?;
    println!("classes: {:?}", matrix.classes());
    println!("confusion matrix:\n{:?}", matrix.counts());

    let report = classification_report(test.targets(), predictions.view())?;
    for metrics in report.entries() {
        println!(
            "{}: precision {:.3}, recall {:.3}, f1 {:.3}",
            metrics.label(),
            metrics.precision(),
            metrics.recall(),
            metrics.f1()
        );
    }
    Ok(())
}

#[cfg(not(feature = "csv"))]
fn main() {
    println!("Run with: cargo run --example iris_classification --features csv");
}
