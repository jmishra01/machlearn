//! Standardizing features and restoring their original units.

use machlearn::{Result, StandardScaler};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    let scaler = StandardScaler::default().fit(records.view())?;
    let standardized = scaler.transform(records.view())?;
    let restored = scaler.inverse_transform(standardized.view())?;

    println!("learned means: {:?}", scaler.mean());
    println!("learned scales: {:?}", scaler.scale());
    println!("standardized records:\n{standardized:?}");

    for (actual, expected) in restored.iter().zip(records.iter()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    Ok(())
}
