# MachLearn

MachLearn is an early-stage, CPU-first machine-learning library written in
Rust. The initial goal is a dependable classical machine-learning API for
dense tabular data.

Dense numerical solvers are isolated behind an internal `faer` backend; public
data and estimator APIs remain based on `ndarray` and MachLearn types.

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
- deterministic cross-validated grid search and ranking;
- optional `serde` and deterministic Rayon model-selection features;
- ordinary least-squares linear regression backed by column-pivoted QR;
- L2-regularized ridge regression with an unpenalized intercept;
- binary logistic regression with deterministic classes and probabilities;
- one-vs-rest multiclass logistic regression with normalized probabilities;
- configurable convergence reporting and stopping criteria for iterative solvers;
- k-nearest-neighbors classification and regression with uniform or distance weighting;
- Gaussian Naive Bayes classification with configurable variance smoothing;
- CART-style decision-tree classification and regression with feature importances;
- bootstrap-aggregated random forests with per-split feature subsampling;
- k-means clustering with k-means++ or random initialization;
- principal component analysis with explained-variance reporting;
- a versioned model-serialization envelope, and optional CSV dataset loading.

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

Property-based tests for cross-cutting numerical invariants (scaler
centering, split completeness, metric bounds, PCA orthonormality, and more)
live in `tests/property_invariants.rs`.

Benchmarks for representative fit/predict paths live in `benches/estimators.rs`:

```text
cargo bench
cargo bench -- --save-baseline before
# ...make a change...
cargo bench -- --baseline before
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

Runnable examples for datasets, splitting, preprocessing, every estimator,
custom models, serialization, CSV loading, and parallel batches are indexed
in `examples/README.md`. For two complete, worked examples on real
(non-synthetic) datasets — classifying Iris flower species and predicting
car fuel economy — see [`docs/user-guide.md`](docs/user-guide.md).

```text
cargo run --example dataset
cargo run --example standard_scaler
cargo test --examples --all-features
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
