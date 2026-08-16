# MachLearn examples

Each example is an independent Cargo target and can be run from the project
root.

| Example            | Demonstrates                                | Command                                                    |
|--------------------|---------------------------------------------|------------------------------------------------------------|
| `dataset`          | Validated datasets and typed targets        | `cargo run --example dataset`                              |
| `train_test_split` | Ordered and deterministic shuffled splits   | `cargo run --example train_test_split`                     |
| `standard_scaler`  | Standardization and inverse transformation  | `cargo run --example standard_scaler`                      |
| `min_max_scaler`   | Default and custom output ranges            | `cargo run --example min_max_scaler`                       |
| `pipeline`         | Ordered fitting without data leakage        | `cargo run --example pipeline`                             |
| `custom_model`     | Implementing the `Fit` and `Predict` traits | `cargo run --example custom_model`                         |
| `serde_model`      | Serializing a fitted transformer            | `cargo run --example serde_model --features serde`         |
| `parallel_batches` | Transforming independent batches with Rayon | `cargo run --example parallel_batches --features parallel` |

Compile every example with every optional feature:

```text
cargo test --examples --all-features
```
