//! Encoding categorical targets for numerical models.

use machlearn::{LabelEncoder, Result};
use ndarray::array;

fn main() -> Result<()> {
    let labels = array!["spam", "ham", "spam", "unknown"];
    let fitted = LabelEncoder::new().fit(labels.view())?;
    let encoded = fitted.transform(labels.view())?;
    let restored = fitted.inverse_transform(encoded.view())?;

    println!("classes: {:?}", fitted.classes());
    println!("encoded labels: {encoded:?}");
    println!("restored labels: {restored:?}");
    Ok(())
}
