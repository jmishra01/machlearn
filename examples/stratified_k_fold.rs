//! Create deterministic cross-validation folds while preserving class balance.

use machlearn::{Result, StratifiedKFold};

fn main() -> Result<()> {
    let targets = [
        "cat", "cat", "cat", "cat", "dog", "dog", "dog", "dog", "dog",
    ];
    let folds = StratifiedKFold::new(3)?
        .with_shuffle(true)
        .with_seed(7)
        .split(&targets)?;

    for (index, fold) in folds.iter().enumerate() {
        let test_labels: Vec<_> = fold
            .test_indices()
            .iter()
            .map(|&sample| targets[sample])
            .collect();
        println!("fold {}", index + 1);
        println!("  train: {:?}", fold.train_indices());
        println!("  test:  {:?} => {test_labels:?}", fold.test_indices());
    }
    Ok(())
}
