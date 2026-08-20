# MachLearn development tracker

Last updated: 2026-08-20

Status markers:

- `[x]` complete
- `[-]` in progress
- `[ ]` pending

## 1. Project foundation

- [x] Create the Rust 2024 library crate
- [x] Define `Fit`, `Predict`, and `Transform` API contracts
- [x] Add a validated dense `Dataset<Target>` type
- [x] Add structured public errors
- [x] Implement deterministic train/test splitting
- [x] Add optional `serde` and parallel feature flags
- [x] Enable strict formatting and Clippy checks
- [x] Add runnable examples for every currently exposed capability
- [x] Choose and add the project license files
- [x] Initialize a Git repository and continuous integration

## 2. Preprocessing

- [x] Implement `StandardScaler`
- [x] Implement `MinMaxScaler`
- [x] Add fitted transformer composition through `Pipeline`
- [x] Implement categorical label encoding
- [x] Define and implement a missing-value policy
- [x] Add train-only fitting examples that demonstrate leakage prevention

## 3. Metrics

- [x] Mean squared error, root mean squared error, and mean absolute error
- [x] R-squared score
- [x] Accuracy and confusion matrix
- [x] Precision, recall, and F1 score
- [x] Binary log loss and ROC AUC
- [x] Define behavior for empty inputs, zero divisions, and invalid probabilities

## 4. Model selection

- [x] K-fold cross-validation
- [x] Stratified K-fold cross-validation
- [x] Cross-validation scoring
- [x] Parameter grid representation
- [x] Grid search
- [x] Optional parallel fold and parameter evaluation

## 5. Linear models

- [x] Add `faer` as the internal numerical solver backend
- [x] Ordinary least-squares regression using QR or SVD
- [x] Ridge regression
- [x] Binary logistic regression
- [x] Multiclass logistic regression
- [x] Convergence reports and configurable stopping criteria

## 6. Additional supervised models

- [x] K-nearest-neighbour classifier and regressor
- [x] Gaussian Naive Bayes
- [x] Decision-tree classifier and regressor
- [x] Random-forest classifier and regressor
- [x] Feature importance reporting

## 7. Unsupervised learning

- [x] K-means and k-means++ initialization
- [x] Principal component analysis
- [x] Explained-variance reporting

## 8. Persistence and integrations

- [x] Design a versioned model serialization envelope
- [x] Model round-trip tests with the `serde` feature
- [x] Optional CSV dataset loading
- [x] Evaluate Arrow or Polars interoperability (see `docs/interop-evaluation.md`)
- [x] Evaluate Python bindings with PyO3 (see `docs/interop-evaluation.md`)
- [x] Evaluate WebAssembly support (see `docs/interop-evaluation.md`)

## 9. Quality and release gates

- [x] Reference-result tests against established implementations
- [x] Property tests for every numerical invariant
- [x] Criterion benchmark suite and saved baselines
- [x] Test default, no-default, and all-feature configurations in CI
- [x] Audit dependencies and licenses (see `docs/dependency-license-audit.md`)
- [x] Complete API documentation and runnable examples
- [x] Define minimum-supported-Rust-version testing
- [ ] Publish the `0.1.0` release

## 10. Regularized and additional linear models

- [x] Lasso regression (L1) via coordinate descent
- [x] Elastic Net regression (combined L1/L2)
- [x] Linear discriminant analysis

## 11. Boosting and model-agnostic evaluation

- [x] Gradient-boosted decision trees (classifier and regressor)
- [x] AdaBoost classifier
- [x] Model-agnostic permutation feature importance

## 12. Naive Bayes variants and additional clustering

- [x] Multinomial Naive Bayes
- [x] Bernoulli Naive Bayes
- [x] DBSCAN density-based clustering
- [x] Gaussian mixture models via expectation-maximization

## 13. Preprocessing expansion

- [ ] One-hot / dummy encoding for categorical features
- [ ] Polynomial and interaction feature generation
- [ ] Variance-threshold and univariate feature selection

## 14. Model selection and evaluation expansion

- [ ] Multiclass log loss and one-vs-rest ROC AUC
- [ ] Randomized hyperparameter search
- [ ] Learning-curve and validation-curve utilities

## 15. Release and ecosystem

- [ ] Revisit Arrow/Polars, PyO3, or WebAssembly bindings if a concrete consumer emerges (see `docs/interop-evaluation.md`)
- [ ] A user guide beyond the README, with worked examples on real (non-synthetic) datasets

`Publish the 0.1.0 release` is tracked once, in section 9, once sections 10-14 below are as far along as the maintainer wants before a first release.

## Completed milestone exit criteria (sections 1-9)

The original library roadmap is complete:

- `faer` is isolated behind an internal numerical-solver module;
- every estimator matches an independent or `scikit-learn` reference solution where one exists;
- classifiers expose stable, normalized probability predictions; iterative solvers report convergence and honor stopping criteria;
- fitted models reject incompatible or non-finite prediction inputs;
- strict Clippy, default-feature, no-default-feature, and all-feature tests pass locally and in CI;
- property tests cover cross-cutting numerical invariants, and a Criterion benchmark suite exists;
- dependency licenses are audited, the MSRV is verified, and the crate has a chosen license.

Publishing `0.1.0` to crates.io is the one remaining step, gated on the
maintainer's own account and judgment call on API stability.

## Next milestone exit criteria

Sections 10-14 are complete when:

- every new estimator or transformer follows the existing `Fit`/`Predict`/`Transform` conventions and `Dataset` API;
- every new algorithm has a reference-checked test (against `scikit-learn` or an independent closed-form derivation) plus structured-error and edge-case coverage;
- new numerical routines get at least one property test for their core invariant;
- strict Clippy, `cargo fmt --check`, and default/no-default/all-feature test runs stay green throughout.
