//! Replacing missing features before constructing a dataset.

use machlearn::{Dataset, Result, SimpleImputer};
use ndarray::array;

fn main() -> Result<()> {
    // NaN is the only missing-value marker. Infinity is always invalid.
    let raw_records = array![[1.0, f64::NAN], [3.0, 2.0], [5.0, 4.0]];
    let targets = array![0, 1, 1];

    let imputer = SimpleImputer::mean().fit(raw_records.view())?;
    let clean_records = imputer.transform(raw_records.view())?;
    let dataset = Dataset::new(clean_records, targets)?;

    println!("learned fill values: {:?}", imputer.fill_values());
    println!("clean dataset records:\n{:?}", dataset.records());
    Ok(())
}
