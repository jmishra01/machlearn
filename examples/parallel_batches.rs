//! Transforming independent data batches in parallel.

#[cfg(feature = "parallel")]
fn main() -> machlearn::Result<()> {
    use machlearn::StandardScaler;
    use ndarray::array;
    use rayon::prelude::*;

    let training = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    let scaler = StandardScaler::default().fit(training.view())?;
    let batches = [
        array![[2.0, 15.0], [4.0, 25.0]],
        array![[6.0, 35.0], [8.0, 45.0]],
    ];

    let transformed: machlearn::Result<Vec<_>> = batches
        .par_iter()
        .map(|batch| scaler.transform(batch.view()))
        .collect();

    for (index, batch) in transformed?.iter().enumerate() {
        println!("batch {index}:\n{batch:?}");
    }
    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn main() {
    println!("Run with: cargo run --example parallel_batches --features parallel");
}
