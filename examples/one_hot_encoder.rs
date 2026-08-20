//! One-hot and dummy encoding a categorical feature column.

use machlearn::{OneHotEncoder, Result};
use ndarray::array;

fn main() -> Result<()> {
    let labels = array!["red", "green", "blue", "green"];

    let one_hot = OneHotEncoder::new().fit(labels.view())?;
    println!("classes: {:?}", one_hot.classes());
    println!("one-hot: {:?}", one_hot.transform(labels.view())?);

    let dummy = OneHotEncoder::new()
        .with_drop_first(true)
        .fit(labels.view())?;
    println!("dummy (drop first): {:?}", dummy.transform(labels.view())?);
    Ok(())
}
