//! Training a model in one process and loading it in another via a
//! `bincode`-encoded file on disk — the shape of a typical train/serve split.

#[cfg(feature = "bincode")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use machlearn::{Dataset, FittedLinearRegression, LinearRegression, ModelEnvelope};
    use ndarray::array;

    // --- Training side: fit a model and persist it. ---
    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0])?;
    let model = LinearRegression::new().fit(&dataset)?;

    let path = std::env::temp_dir().join("machlearn_linear_regression.bin");
    ModelEnvelope::new(model).save_bincode_file(&path)?;
    println!("wrote model to {}", path.display());

    // --- Serving side: load the file back and predict, without retraining. ---
    let restored: FittedLinearRegression =
        ModelEnvelope::load_bincode_file(&path)?.into_model()?;
    let predictions = restored.predict(array![[4.0], [5.0]].view())?;
    println!("predictions: {predictions:?}");

    std::fs::remove_file(&path)?;
    Ok(())
}

#[cfg(not(feature = "bincode"))]
fn main() {
    println!("Run with: cargo run --example model_persistence_bincode --features bincode");
}
