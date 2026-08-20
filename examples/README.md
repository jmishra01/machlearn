# MachLearn examples

Each example is an independent Cargo target and can be run from the project
root.

| Example                  | Demonstrates                                          | Command                                                    |
|--------------------------|-------------------------------------------------------|------------------------------------------------------------|
| `dataset`                | Validated datasets and typed targets                  | `cargo run --example dataset`                              |
| `train_test_split`       | Ordered and deterministic shuffled splits             | `cargo run --example train_test_split`                     |
| `standard_scaler`        | Standardization and inverse transformation            | `cargo run --example standard_scaler`                      |
| `min_max_scaler`         | Default and custom output ranges                      | `cargo run --example min_max_scaler`                       |
| `label_encoder`          | Reversible categorical target encoding                | `cargo run --example label_encoder`                        |
| `simple_imputer`         | Explicit mean/median/constant imputation              | `cargo run --example simple_imputer`                       |
| `pipeline`               | Ordered fitting without data leakage                  | `cargo run --example pipeline`                             |
| `regression_metrics`     | MSE, RMSE, MAE, and R-squared                         | `cargo run --example regression_metrics`                   |
| `classification_metrics` | Accuracy, confusion matrix, precision, recall, and F1 | `cargo run --example classification_metrics`               |
| `probability_metrics`    | Binary log loss and tie-aware ROC AUC                 | `cargo run --example probability_metrics`                  |
| `k_fold`                 | Balanced ordered or seeded shuffled folds             | `cargo run --example k_fold`                               |
| `stratified_k_fold`      | Class-balanced ordered or shuffled folds               | `cargo run --example stratified_k_fold`                    |
| `cross_validation`       | Independent per-fold model fitting and scoring          | `cargo run --example cross_validation`                     |
| `parameter_grid`         | Deterministic hyperparameter Cartesian products          | `cargo run --example parameter_grid`                       |
| `grid_search`            | Cross-validated hyperparameter ranking                    | `cargo run --example grid_search`                          |
| `parallel_cross_validation` | Concurrent deterministic fold scoring                 | `cargo run --example parallel_cross_validation --features parallel` |
| `parallel_grid_search`   | Concurrent deterministic grid and fold evaluation         | `cargo run --example parallel_grid_search --features parallel` |
| `linear_regression`      | Ordinary least-squares fitting and prediction            | `cargo run --example linear_regression`                    |
| `ridge_regression`       | L2-regularized regression with intercept handling        | `cargo run --example ridge_regression`                     |
| `logistic_regression`    | Binary classification and class probabilities             | `cargo run --example logistic_regression`                  |
| `multiclass_logistic_regression` | One-vs-rest classification and normalized probabilities | `cargo run --example multiclass_logistic_regression`    |
| `knn_classifier`         | Voting among nearby points with uniform and distance weighting | `cargo run --example knn_classifier`               |
| `knn_regressor`          | Averaging nearby targets with uniform and distance weighting | `cargo run --example knn_regressor`                  |
| `gaussian_naive_bayes`   | Per-class Gaussian fitting and normalized probabilities | `cargo run --example gaussian_naive_bayes`               |
| `decision_tree_classifier` | Gini-minimizing splits, leaf probabilities, and feature importances | `cargo run --example decision_tree_classifier` |
| `decision_tree_regressor` | Variance-minimizing splits, depth-limited predictions, and feature importances | `cargo run --example decision_tree_regressor` |
| `random_forest_classifier` | Bootstrap-aggregated trees, averaged leaf probabilities, and feature importances | `cargo run --example random_forest_classifier` |
| `random_forest_regressor` | Bootstrap-aggregated trees, averaged predictions, and feature importances | `cargo run --example random_forest_regressor` |
| `kmeans`                 | k-means++ initialization, cluster assignment, and inertia | `cargo run --example kmeans`                            |
| `pca`                    | Dimensionality reduction and explained-variance reporting | `cargo run --example pca`                               |
| `custom_model`           | Implementing the `Fit` and `Predict` traits           | `cargo run --example custom_model`                         |
| `serde_model`            | Serializing a fitted transformer                      | `cargo run --example serde_model --features serde`         |
| `parallel_batches`       | Transforming independent batches with Rayon           | `cargo run --example parallel_batches --features parallel` |

Compile every example with every optional feature:

```text
cargo test --examples --all-features
```
