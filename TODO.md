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

- [x] One-hot / dummy encoding for categorical features
- [x] Polynomial and interaction feature generation
- [x] Variance-threshold and univariate feature selection

## 14. Model selection and evaluation expansion

- [x] Multiclass log loss and one-vs-rest ROC AUC
- [x] Randomized hyperparameter search
- [x] Learning-curve and validation-curve utilities

## 15. Release and ecosystem

- [x] Split the crate into a Cargo workspace (`machlearn-core`, `-metrics`,
      `-preprocessing`, `-linear`, `-trees`, `-unsupervised`,
      `-model-selection`, with `machlearn` as a re-exporting facade) — the
      prerequisite `docs/interop-evaluation.md` calls for before adding a
      PyO3 binding crate as a new workspace member
- [x] Arrow interoperability (`arrow` feature flag on `machlearn-preprocessing`,
      backed by `arrow-array`/`arrow-schema` rather than the heavier umbrella
      `arrow` crate): `arrays_from_record_batch` in the `io` module dispatches
      per Arrow numeric type into a dense `f64` `Array2`, mapping Arrow nulls
      to `NaN` and returning raw arrays (not a `Dataset`, which rejects `NaN`)
      so callers impute with `SimpleImputer` first, exactly as
      `docs/interop-evaluation.md` recommended; targets `arrow-rs` directly,
      no bespoke Polars-specific path, since Polars is itself Arrow-backed
- [x] PyO3 Python bindings — starter crate (`python/machlearn-python`):
      `Dataset`, `train_test_split`, `LinearRegression`, and `KMeans` bound
      via `numpy`/`ndarray` conversion helpers, with a `maturin`
      `pyproject.toml` and a `machlearn.pyi` type stub. Verified end-to-end
      with `maturin develop` + a Python smoke script. Deliberately kept
      **outside** the Cargo workspace (`exclude = ["python"]` in the root
      `Cargo.toml`), not a workspace member as originally scoped: PyO3's
      `extension-module` feature makes a plain `cargo test` unittest binary
      fail to link, which would otherwise break every `cargo
      build/test/clippy --workspace` command; see `python/README.md`.
- [x] Extend the PyO3 bindings to cover a full binary-classification
      workflow: `LogisticRegression` (fit takes raw integer-labeled
      `records`/`targets` arrays rather than a `Dataset`, since `f64` doesn't
      implement the `Ord` bound `LogisticRegression::fit` requires),
      `StandardScaler`, and `accuracy_score`/`mean_squared_error`/`r2_score`.
      Verified end-to-end with `maturin develop` + a Python smoke script
      exercising scale → fit → predict → evaluate.
- [x] Extend the PyO3 bindings to tree-based estimators:
      `DecisionTreeRegressor`, `DecisionTreeClassifier`,
      `RandomForestRegressor`, `RandomForestClassifier` (classifiers use the
      same raw-integer-array `fit` shape as `LogisticRegression`, for the
      same `Ord`-bound reason). Verified end-to-end with `maturin develop` +
      a Python smoke script for both regression and classification.
- [x] Extend the PyO3 bindings to `KNeighborsRegressor`/`Classifier` and
      `GaussianNaiveBayes` (multi-class, unlike binary-only
      `LogisticRegression`; exposes `.classes`/`.predict_proba`). Verified
      end-to-end with `maturin develop` + a Python smoke script.
- [x] Extend the PyO3 bindings to the rest of the linear model family:
      `RidgeRegression`, `LassoRegression`, `ElasticNetRegression`
      (`Dataset`-based), and `LinearDiscriminantAnalysis` (raw-array-based,
      multi-class). Verified end-to-end with `maturin develop` + a Python
      smoke script.
- [x] Extend the PyO3 bindings to boosting ensembles:
      `GradientBoostingRegressor` (`Dataset`-based),
      `GradientBoostingClassifier` and `AdaBoostClassifier` (raw-array-based,
      binary, like `LogisticRegression`). Verified end-to-end with `maturin
      develop` + a Python smoke script.
- [x] Extend the PyO3 bindings to clustering/decomposition: `DBSCAN` (no
      `predict` for new points, only `.labels()` on the fitted rows, noise
      encoded as `-1`), `GaussianMixture`, and `PCA`. Verified end-to-end
      with `maturin develop` + a Python smoke script.
- [x] Extend the PyO3 bindings to `MultinomialNaiveBayes` and
      `BernoulliNaiveBayes` (same multi-class, raw-array pattern as
      `GaussianNaiveBayes`). Verified end-to-end with `maturin develop` + a
      Python smoke script.
- [x] Extend the PyO3 bindings to the remaining preprocessing transformers:
      `SimpleImputer`, `PolynomialFeatures`, `LabelEncoder`, `OneHotEncoder`
      (the latter two take a Python `list[str]` rather than a numpy array,
      since labels are categorical values, not features). Verified
      end-to-end with `maturin develop` + a Python smoke script.
- [x] Extend the PyO3 bindings to the remaining metrics: `precision_score`,
      `recall_score`, `f1_score`, `roc_auc_score`, and `confusion_matrix`
      (returns `(counts, classes)` rather than a wrapper class). Verified
      end-to-end with `maturin develop` + a Python smoke script.
- [x] Extend the PyO3 bindings to `KFold`/`StratifiedKFold`
      (`.split(...)` returns `[(train_indices, test_indices), ...]`).
      Verified end-to-end with `maturin develop` + a Python smoke script,
      including a full manual cross-validation loop combining `KFold` with
      `LinearRegression`.
- [ ] `cross_validate`/`grid_search`/`randomized_search`/
      `permutation_importance`/`learning_curve`/`validation_curve` remain
      unbound: each is generic in Rust over an `Estimator: Fit` type
      parameter plus a scorer closure, resolved at compile time, so there is
      no single monomorphic function to bind the way every other estimator
      was. Binding these needs either one wrapper per already-bound
      estimator type, or a callback layer accepting Python callables
      (an estimator factory returning an already-bound PyO3 class, plus a
      scorer function) that calls back into Python per fold — a real design
      change, not a mechanical continuation of the per-type template used
      for everything else. A manual `KFold` + per-fold `fit`/`predict` loop
      (see `python/README.md`) covers the same ground today. With this, every
      other public estimator, transformer, and metric in the Rust crate has
      a PyO3 binding.
- [x] Add a dedicated Python + maturin CI job: `.github/workflows/python.yml`
      (separate from `rust.yml` since `python/` isn't a workspace member),
      running `cargo fmt --check`/`clippy -D warnings` for the binding
      crate, `maturin develop`, and a new `python/tests/test_bindings.py`
      pytest suite (34 tests) covering every bound class/function. Verified
      by running the exact CI sequence locally in a fresh virtualenv.
- [ ] Publish wheels / a PyPI release once the bound surface is broader
- [ ] WebAssembly wrapper crate (`wasm-bindgen`, published npm package) if a
      concrete consumer emerges — the library itself is already wasm-ready
      for non-parallel use at zero further cost (see `docs/interop-evaluation.md`)
- [x] A user guide beyond the README, with worked examples on real (non-synthetic) datasets

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
