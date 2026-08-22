# machlearn (Python bindings)

PyO3 bindings for the [MachLearn](../README.md) Rust machine-learning
library. This is still a **starter binding**, not the full public API. It
currently exposes:

- `Dataset`, `train_test_split`
- `LinearRegression`, `KMeans` (the original starter pair: one regression,
  one clustering estimator)
- `LogisticRegression`, `StandardScaler`, `accuracy_score`,
  `mean_squared_error`, `r2_score` — enough for a full
  load → scale → fit → predict → evaluate binary classification workflow
  (see the example below)
- `DecisionTreeRegressor`/`Classifier`, `RandomForestRegressor`/`Classifier`
  — tree-based estimators, following the same `Dataset`-for-regression /
  raw-integer-array-for-classification split as `LinearRegression`/
  `LogisticRegression`
- `KNeighborsRegressor`/`Classifier`, `GaussianNaiveBayes` — same split
  again; `GaussianNaiveBayes` supports any number of classes (unlike the
  binary-only `LogisticRegression`), exposing `.classes`/`.predict_proba`
- `RidgeRegression`, `LassoRegression`, `ElasticNetRegression` (all
  `Dataset`-based, like `LinearRegression`), and
  `LinearDiscriminantAnalysis` (raw-array-based, multi-class, like
  `GaussianNaiveBayes`)
- `GradientBoostingRegressor` (`Dataset`-based), `GradientBoostingClassifier`
  and `AdaBoostClassifier` (raw-array-based, binary, like
  `LogisticRegression`)
- `DBSCAN` (no `predict` for new points — only `.labels()` for the fitted
  rows, with noise encoded as `-1`), `GaussianMixture`, and `PCA`
- `MultinomialNaiveBayes` and `BernoulliNaiveBayes` (same multi-class,
  raw-array pattern as `GaussianNaiveBayes`)
- `SimpleImputer` (`.mean()`/`.median()`/`.constant()` static constructors,
  matching the Rust API's own shape — there is no `SimpleImputer()`),
  `PolynomialFeatures`, `LabelEncoder`, and `OneHotEncoder`.
  `LabelEncoder`/`OneHotEncoder` take a Python `list[str]` rather than a
  numpy array: labels are categorical values, not features, and
  `Vec<String>` converts directly from `list[str]` without needing numpy's
  less ergonomic string-array support.
- `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, and
  `confusion_matrix` (returns `(counts, classes)` rather than a wrapper
  class)
- `KFold`/`StratifiedKFold` — `.split(...)` returns
  `[(train_indices, test_indices), ...]`; combine with any bound estimator
  by indexing your own arrays per fold (see the example below). This is as
  far as model-selection binding goes in this phase:
  `cross_validate`/`grid_search`/`randomized_search`/`permutation_importance`
  are generic in Rust over an `Estimator: Fit` type parameter plus a scorer
  closure, resolved at compile time — there's no monomorphic function to
  bind without either one wrapper per already-bound estimator type, or a
  callback layer that calls back into Python per fold. Left for a future
  phase; the manual `KFold` + per-fold `fit`/`predict` loop below covers the
  same ground today.

`LogisticRegression.fit` takes raw `records`/`targets` arrays rather than a
`Dataset`, because `machlearn::LogisticRegression::fit` requires a label
type implementing `Ord` (for a deterministic negative/positive class
order), which `f64` — the label type `Dataset` is bound to here — does not
implement (`NaN` breaks a total order). Integer labels sidestep that
without introducing a second public `Dataset` type in this phase.

Every estimator, preprocessing transformer, and metric in the Rust crate is
now bound, following the template established above. The one deliberate gap
is the higher-order model-selection functions noted above
(`cross_validate`/`grid_search`/`randomized_search`/`permutation_importance`),
which need a different design (see above) rather than the per-type template
every other binding followed.

This crate is deliberately **not** a member of the main Cargo workspace at
the repository root (see the `exclude` entry in the root `Cargo.toml`):
PyO3's `extension-module` feature makes a plain `cargo test` unittest
harness binary fail to link/run, since that feature intentionally omits
linking against `libpython` (a real Python interpreter is expected to load
the module instead). Keeping this crate external means the main workspace's
`cargo build/test/clippy --workspace` commands are unaffected by that.

## Building and installing locally

Requires a Python virtualenv (or conda environment) active, and
[`maturin`](https://www.maturin.rs/) installed in it:

```bash
cd python
maturin develop --release
python3 -c "import machlearn; print(machlearn.LinearRegression())"
```

## Tests

`tests/test_bindings.py` (pytest) exercises every bound class/function end to
end against the compiled extension — install it with `maturin develop`
first, then:

```bash
pip install maturin pytest numpy
maturin develop --release
pytest tests -v
```

CI (`.github/workflows/python.yml`) runs `cargo fmt --check`, `cargo clippy
--all-targets -- -D warnings`, and this test suite on every push/PR,
alongside the separate Rust-only workflow — this crate isn't a workspace
member, so it needed its own job rather than folding into `rust.yml`.

`cargo build`/`cargo check` from this directory also work directly (a
`.cargo/config.toml` here sets the macOS linker flag `maturin` would
otherwise apply automatically), which is useful for fast iteration without
rebuilding the wheel — but only `maturin develop`/`maturin build` produce an
actually importable module.

## Example: regression

```python
import numpy as np
import machlearn

dataset = machlearn.Dataset(
    np.array([[1.0], [2.0], [3.0], [4.0]]),
    np.array([10.0, 20.0, 30.0, 40.0]),
)
train, test = machlearn.train_test_split(dataset, 0.25, seed=7)

model = machlearn.LinearRegression()
model.fit(train)
print(model.predict(test.records))
```

## Example: classification workflow

```python
import numpy as np
import machlearn

records = np.array([[0.1, 1.0], [0.2, 1.1], [5.0, 6.0], [5.2, 6.3]])
targets = np.array([0, 0, 1, 1], dtype=np.int64)

scaler = machlearn.StandardScaler()
scaled = scaler.fit_transform(records)

model = machlearn.LogisticRegression()
model.fit(scaled, targets)
predictions = model.predict(scaled)

print("accuracy:", machlearn.accuracy_score(targets, predictions))
```

## Example: manual cross-validation with KFold

```python
import numpy as np
import machlearn

records = np.arange(9.0).reshape(9, 1)
targets = records.flatten() * 2 + 1

scores = []
for train_idx, test_idx in machlearn.KFold(n_splits=3, shuffle=True, seed=7).split(9):
    model = machlearn.LinearRegression()
    model.fit(machlearn.Dataset(records[train_idx], targets[train_idx]))
    predictions = model.predict(records[test_idx])
    scores.append(machlearn.r2_score(targets[test_idx], predictions))

print("per-fold R2:", scores)
```
