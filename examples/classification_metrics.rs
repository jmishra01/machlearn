//! Evaluating categorical predictions with accuracy and a confusion matrix.

use machlearn::{
    Averaging, ClassificationMetricOptions, Result, accuracy_score, classification_report,
    confusion_matrix, f1_score_with_options, precision_score_with_options,
    recall_score_with_options,
};
use ndarray::array;

fn main() -> Result<()> {
    let actual = array!["cat", "dog", "cat", "bird"];
    let predicted = array!["cat", "cat", "dog", "bird"];

    let accuracy = accuracy_score(actual.view(), predicted.view())?;
    let matrix = confusion_matrix(actual.view(), predicted.view())?;
    let report = classification_report(actual.view(), predicted.view())?;
    let macro_average = ClassificationMetricOptions::new().with_averaging(Averaging::Macro);

    println!("accuracy: {accuracy:.3}");
    println!(
        "macro precision: {:.3}",
        precision_score_with_options(actual.view(), predicted.view(), macro_average)?
    );
    println!(
        "macro recall: {:.3}",
        recall_score_with_options(actual.view(), predicted.view(), macro_average)?
    );
    println!(
        "macro F1: {:.3}",
        f1_score_with_options(actual.view(), predicted.view(), macro_average)?
    );
    println!("class order: {:?}", matrix.classes());
    println!("rows=actual, columns=predicted:\n{:?}", matrix.counts());
    for entry in report.entries() {
        println!(
            "{:?}: precision={:.3}, recall={:.3}, F1={:.3}, support={}",
            entry.label(),
            entry.precision(),
            entry.recall(),
            entry.f1(),
            entry.support()
        );
    }
    Ok(())
}
