//! Creating reproducible training and test partitions.

use machlearn::{Dataset, Result, SplitOptions, train_test_split};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![0, 1, 2, 3, 4, 5],
    )?;

    let options = SplitOptions::new(1.0 / 3.0).with_seed(2026);
    let (train, test) = train_test_split(&dataset, options)?;

    println!("training targets: {:?}", train.targets());
    println!("test targets: {:?}", test.targets());

    // Reusing the same seed produces the same partition.
    let (repeated_train, repeated_test) = train_test_split(&dataset, options)?;
    assert_eq!(train, repeated_train);
    assert_eq!(test, repeated_test);

    Ok(())
}
