//! Evaluating multiclass probability predictions.

use machlearn::{Result, multiclass_log_loss, roc_auc_score_ovr};
use ndarray::array;

fn main() -> Result<()> {
    let actual = array!["a", "b", "c", "a", "b", "c"];
    let probabilities = array![
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
    ];
    let classes = ["a", "b", "c"];

    let log_loss = multiclass_log_loss(actual.view(), probabilities.view(), &classes)?;
    let roc_auc = roc_auc_score_ovr(actual.view(), probabilities.view(), &classes)?;

    println!("multiclass log loss:   {log_loss:.6}");
    println!("one-vs-rest ROC AUC:   {roc_auc:.6}");
    Ok(())
}
