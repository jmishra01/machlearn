//! Evaluating binary positive-class probabilities.

use machlearn::{Result, binary_log_loss, roc_auc_score};
use ndarray::array;

fn main() -> Result<()> {
    let actual = array!["no", "no", "yes", "yes"];
    let positive_probabilities = array![0.1, 0.4, 0.35, 0.8];

    let log_loss = binary_log_loss(actual.view(), positive_probabilities.view(), &"yes")?;
    let roc_auc = roc_auc_score(actual.view(), positive_probabilities.view(), &"yes")?;

    println!("binary log loss: {log_loss:.6}");
    println!("ROC AUC:         {roc_auc:.6}");
    Ok(())
}
