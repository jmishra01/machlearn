//! Wrapping a fitted model with a version tag before serializing it.

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use machlearn::{Dataset, FittedLinearRegression, LinearRegression, ModelEnvelope};
    use ndarray::array;

    let dataset = Dataset::new(array![[1.0], [2.0], [3.0]], array![2.0, 4.0, 6.0])?;
    let model = LinearRegression::new().fit(&dataset)?;

    let envelope = ModelEnvelope::new(model);
    println!("envelope version: {}", envelope.version());

    let json = serde_json::to_string_pretty(&envelope)?;
    println!("serialized envelope:\n{json}");

    let restored: ModelEnvelope<FittedLinearRegression> = serde_json::from_str(&json)?;
    let restored_model = restored.into_model()?;
    println!("restored coefficients: {:?}", restored_model.coefficients());
    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() {
    println!("Run with: cargo run --example model_envelope --features serde");
}
