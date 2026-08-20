//! Scoring features against a target and keeping only the strongest ones.

use machlearn::{Result, SelectKBest, f_classif};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [1.0, 10.0],
        [2.0, 9.0],
        [1.5, 11.0],
        [8.0, 1.0],
        [9.0, 2.0],
        [8.5, 0.5],
    ];
    let targets = array!["a", "a", "a", "b", "b", "b"];

    let fitted = SelectKBest::new(1).fit(records.view(), targets.view(), f_classif)?;

    println!("scores: {:?}", fitted.scores());
    println!("selected columns: {:?}", fitted.selected_indices());
    println!("selected:\n{:?}", fitted.transform(records.view())?);
    Ok(())
}
