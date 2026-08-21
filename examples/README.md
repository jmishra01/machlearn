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
| `one_hot_encoder`        | One-hot and dummy (drop-first) categorical feature encoding | `cargo run --example one_hot_encoder`                |
| `polynomial_features`    | Polynomial and interaction-only feature expansion      | `cargo run --example polynomial_features`                  |
| `variance_threshold`     | Removing exactly constant (or low-variance) features   | `cargo run --example variance_threshold`                   |
| `select_k_best`          | Univariate F-test scoring and top-k feature selection  | `cargo run --example select_k_best`                        |
| `simple_imputer`         | Explicit mean/median/constant imputation              | `cargo run --example simple_imputer`                       |
| `pipeline`               | Ordered fitting without data leakage                  | `cargo run --example pipeline`                             |
| `regression_metrics`     | MSE, RMSE, MAE, and R-squared                         | `cargo run --example regression_metrics`                   |
| `classification_metrics` | Accuracy, confusion matrix, precision, recall, and F1 | `cargo run --example classification_metrics`               |
| `probability_metrics`    | Binary log loss and tie-aware ROC AUC                 | `cargo run --example probability_metrics`                  |
| `multiclass_probability_metrics` | Multiclass log loss and macro-averaged one-vs-rest ROC AUC | `cargo run --example multiclass_probability_metrics` |
| `k_fold`                 | Balanced ordered or seeded shuffled folds             | `cargo run --example k_fold`                               |
| `stratified_k_fold`      | Class-balanced ordered or shuffled folds               | `cargo run --example stratified_k_fold`                    |
| `cross_validation`       | Independent per-fold model fitting and scoring          | `cargo run --example cross_validation`                     |
| `parameter_grid`         | Deterministic hyperparameter Cartesian products          | `cargo run --example parameter_grid`                       |
| `grid_search`            | Cross-validated hyperparameter ranking                    | `cargo run --example grid_search`                          |
| `randomized_search`      | Cross-validated ranking over a bounded random draw of a grid | `cargo run --example randomized_search`                 |
| `parallel_cross_validation` | Concurrent deterministic fold scoring                 | `cargo run --example parallel_cross_validation --features parallel` |
| `parallel_grid_search`   | Concurrent deterministic grid and fold evaluation         | `cargo run --example parallel_grid_search --features parallel` |
| `parallel_randomized_search` | Concurrent randomized search and fold evaluation      | `cargo run --example parallel_randomized_search --features parallel` |
| `learning_curve`         | Train/test scores as training-set size grows             | `cargo run --example learning_curve`                       |
| `validation_curve`       | Train/test scores as a hyperparameter is swept            | `cargo run --example validation_curve`                     |
| `linear_regression`      | Ordinary least-squares fitting and prediction            | `cargo run --example linear_regression`                    |
| `ridge_regression`       | L2-regularized regression with intercept handling        | `cargo run --example ridge_regression`                     |
| `lasso_regression`       | L1-regularized regression and sparse coefficients        | `cargo run --example lasso_regression`                     |
| `elastic_net_regression` | Combined L1/L2 regression compared against Lasso         | `cargo run --example elastic_net_regression`                |
| `logistic_regression`    | Binary classification and class probabilities             | `cargo run --example logistic_regression`                  |
| `multiclass_logistic_regression` | One-vs-rest classification and normalized probabilities | `cargo run --example multiclass_logistic_regression`    |
| `linear_discriminant_analysis` | Shared-covariance Gaussian discriminant scores and probabilities | `cargo run --example linear_discriminant_analysis` |
| `knn_classifier`         | Voting among nearby points with uniform and distance weighting | `cargo run --example knn_classifier`               |
| `knn_regressor`          | Averaging nearby targets with uniform and distance weighting | `cargo run --example knn_regressor`                  |
| `gaussian_naive_bayes`   | Per-class Gaussian fitting and normalized probabilities | `cargo run --example gaussian_naive_bayes`               |
| `multinomial_naive_bayes` | Smoothed count-frequency likelihoods and normalized probabilities | `cargo run --example multinomial_naive_bayes` |
| `bernoulli_naive_bayes`  | Binarized presence/absence likelihoods and normalized probabilities | `cargo run --example bernoulli_naive_bayes` |
| `decision_tree_classifier` | Gini-minimizing splits, leaf probabilities, and feature importances | `cargo run --example decision_tree_classifier` |
| `decision_tree_regressor` | Variance-minimizing splits, depth-limited predictions, and feature importances | `cargo run --example decision_tree_regressor` |
| `random_forest_classifier` | Bootstrap-aggregated trees, averaged leaf probabilities, and feature importances | `cargo run --example random_forest_classifier` |
| `random_forest_regressor` | Bootstrap-aggregated trees, averaged predictions, and feature importances | `cargo run --example random_forest_regressor` |
| `gradient_boosting_regressor` | Sequential residual-fitting trees minimizing squared error | `cargo run --example gradient_boosting_regressor` |
| `gradient_boosting_classifier` | Sequential residual-fitting trees minimizing log loss | `cargo run --example gradient_boosting_classifier` |
| `adaboost_classifier`    | Reweighting samples across boosted decision stumps (discrete SAMME) | `cargo run --example adaboost_classifier` |
| `permutation_importance` | Ranking features by shuffled-column score degradation, for any model | `cargo run --example permutation_importance` |
| `kmeans`                 | k-means++ initialization, cluster assignment, and inertia | `cargo run --example kmeans`                            |
| `dbscan`                 | Density-connected clustering with automatic noise detection | `cargo run --example dbscan`                          |
| `gaussian_mixture`       | Expectation-maximization fitting and soft cluster membership | `cargo run --example gaussian_mixture`               |
| `pca`                    | Dimensionality reduction and explained-variance reporting | `cargo run --example pca`                               |
| `custom_model`           | Implementing the `Fit` and `Predict` traits           | `cargo run --example custom_model`                         |
| `serde_model`            | Serializing a fitted transformer                      | `cargo run --example serde_model --features serde`         |
| `model_envelope`         | Version-tagging a fitted model before serializing it  | `cargo run --example model_envelope --features serde`      |
| `csv_dataset`            | Loading a dataset from CSV data                       | `cargo run --example csv_dataset --features csv`            |
| `iris_classification`    | Full worked classification example on real Iris flower data (see the [user guide](../docs/user-guide.md)) | `cargo run --example iris_classification --features csv` |
| `mtcars_regression`      | Full worked regression example on real 1974 car road-test data (see the [user guide](../docs/user-guide.md)) | `cargo run --example mtcars_regression --features csv` |
| `parallel_batches`       | Transforming independent batches with Rayon           | `cargo run --example parallel_batches --features parallel` |

Compile every example with every optional feature:

```text
cargo test --examples --all-features
```
