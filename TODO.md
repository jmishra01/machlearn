# MachLearn development tracker

Last updated: 2026-08-17

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
- [ ] Choose and add the project license files
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
- [ ] Multiclass logistic regression
- [ ] Convergence reports and configurable stopping criteria

## 6. Additional supervised models

- [ ] K-nearest-neighbour classifier and regressor
- [ ] Gaussian Naive Bayes
- [ ] Decision-tree classifier and regressor
- [ ] Random-forest classifier and regressor
- [ ] Feature importance reporting

## 7. Unsupervised learning

- [ ] K-means and k-means++ initialization
- [ ] Principal component analysis
- [ ] Explained-variance reporting

## 8. Persistence and integrations

- [ ] Design a versioned model serialization envelope
- [ ] Model round-trip tests with the `serde` feature
- [ ] Optional CSV dataset loading
- [ ] Evaluate Arrow or Polars interoperability
- [ ] Evaluate Python bindings with PyO3
- [ ] Evaluate WebAssembly support

## 9. Quality and release gates

- [ ] Reference-result tests against established implementations
- [ ] Property tests for every numerical invariant
- [ ] Criterion benchmark suite and saved baselines
- [ ] Test default, no-default, and all-feature configurations in CI
- [ ] Audit dependencies and licenses
- [ ] Complete API documentation and runnable examples
- [ ] Define minimum-supported-Rust-version testing
- [ ] Publish the `0.1.0` release

## Current milestone exit criteria

The linear-model milestone is complete when:

- `faer` is isolated behind an internal numerical-solver module;
- ordinary least-squares and ridge regression match reference solutions;
- binary and multiclass logistic regression expose stable probability predictions;
- iterative solvers report convergence and honor stopping criteria;
- fitted models reject incompatible or non-finite prediction inputs;
- strict Clippy, default-feature tests, and all-feature tests pass.
