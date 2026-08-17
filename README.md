# MachLearn

MachLearn is an early-stage, CPU-first machine-learning library written in
Rust. The initial goal is a dependable classical machine-learning API for
dense tabular data.

The current foundation provides:

- a validated `Dataset` type backed by `ndarray`;
- reusable `Fit`, `Predict`, and `Transform` traits;
- deterministic train/test splitting;
- fitted standard and min-max feature scalers;
- deterministic categorical label encoding;
- explicit missing-value imputation;
- extensible sequential preprocessing pipelines;
- regression and classification evaluation metrics;
- binary probability metrics with strict probability validation;
- deterministic balanced K-fold partitioning;
- deterministic stratified K-fold partitioning;
- independent per-fold cross-validation scoring;
- deterministic hyperparameter-grid expansion;
- optional `serde` and parallel-execution features.

## Example

```rust
use machlearn::{Dataset, SplitOptions, train_test_split};
use ndarray::array;

let dataset = Dataset::new(
    array![[1.0], [2.0], [3.0], [4.0]],
    array![10.0, 20.0, 30.0, 40.0],
)?;

let (train, test) = train_test_split(
    &dataset,
    SplitOptions::default().with_seed(7),
)?;

assert_eq!(train.n_samples() + test.n_samples(), 4);
# Ok::<(), machlearn::MlError>(())
```

## Development

```text
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

The crate is not yet published. Its public API may change while the first
algorithms and pipelines are being implemented.

Development progress and pending milestones are maintained in `TODO.md`.

## Missing-value policy

`NaN` is the only supported missing-value marker. `Dataset`, scalers, and model
inputs reject it by default; first fit and apply `SimpleImputer` to raw arrays.
Positive and negative infinity are invalid everywhere. Mean, median, and finite
constant imputation strategies are available.

## More examples

Runnable examples for datasets, splitting, preprocessing, custom models,
serialization, and parallel batches are indexed in `examples/README.md`.

```text
cargo run --example dataset
cargo run --example standard_scaler
cargo test --examples --all-features
```
