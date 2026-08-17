//! Evaluating categorical predictions with accuracy and a confusion matrix.

use machlearn::{Result, accuracy_score, confusion_matrix};
use ndarray::array;

fn main() -> Result<()> {
    let actual = array!["cat", "dog", "cat", "bird"];
    let predicted = array!["cat", "cat", "dog", "bird"];

    let accuracy = accuracy_score(actual.view(), predicted.view())?;
    let matrix = confusion_matrix(actual.view(), predicted.view())?;

    println!("accuracy: {accuracy:.3}");
    println!("class order: {:?}", matrix.classes());
    println!("rows=actual, columns=predicted:\n{:?}", matrix.counts());
    Ok(())
}
