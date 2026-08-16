//! Mapping features into a configurable range.

use machlearn::{MinMaxScaler, Result};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![[10.0, 2.0], [20.0, 2.0], [30.0, 2.0]];
    let scaler = MinMaxScaler::new(-1.0, 1.0)?.fit(records.view())?;
    let transformed = scaler.transform(records.view())?;
    let restored = scaler.inverse_transform(transformed.view())?;

    println!("observed minima: {:?}", scaler.data_min());
    println!("observed maxima: {:?}", scaler.data_max());
    println!("scaled records:\n{transformed:?}");

    // The constant second column remains finite and maps to the lower bound.
    assert!(
        transformed
            .column(1)
            .iter()
            .all(|value| (*value + 1.0).abs() < 1.0e-12)
    );
    for (actual, expected) in restored.iter().zip(records.iter()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    Ok(())
}
