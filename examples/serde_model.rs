//! Serializing and restoring fitted preprocessing state.

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use machlearn::{FittedStandardScaler, StandardScaler};
    use ndarray::array;

    let records = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    let fitted = StandardScaler::default().fit(records.view())?;

    let json = serde_json::to_string_pretty(&fitted)?;
    println!("serialized fitted scaler:\n{json}");

    let restored: FittedStandardScaler = serde_json::from_str(&json)?;
    assert_eq!(fitted, restored);
    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() {
    println!("Run with: cargo run --example serde_model --features serde");
}
