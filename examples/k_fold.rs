//! Building reproducible K-fold train/test partitions.

use machlearn::{KFold, Result};

fn main() -> Result<()> {
    let folds = KFold::new(3)?
        .with_shuffle(true)
        .with_seed(2026)
        .split(10)?;

    for (index, fold) in folds.iter().enumerate() {
        println!("fold {}", index + 1);
        println!("  train: {:?}", fold.train_indices());
        println!("  test:  {:?}", fold.test_indices());
    }
    Ok(())
}
