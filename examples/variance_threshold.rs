//! Removing exactly constant (and, more generally, low-variance) features.

use machlearn::{Result, VarianceThreshold};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [1.0, 2.0, 0.0, 5.0],
        [1.0, 4.0, 0.0, 3.0],
        [1.0, 6.0, 0.0, 1.0],
        [1.0, 8.0, 0.0, 8.0],
    ];
    let fitted = VarianceThreshold::new().fit(records.view())?;

    println!("variances: {:?}", fitted.variances());
    println!("selected columns: {:?}", fitted.selected_indices());
    println!("filtered:\n{:?}", fitted.transform(records.view())?);
    Ok(())
}
