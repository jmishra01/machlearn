# User guide

The [README](../README.md) covers installation, feature flags, and a
one-estimator quick start. This guide goes further: two complete, worked
examples on real datasets, walked through step by step, plus pointers to
where to go once you outgrow them. Every code block here is runnable — the
full source lives in `examples/` and is exercised in CI, so it stays
correct as the crate changes.

Both datasets are classic, public-domain reference datasets (not synthetic
data generated for the occasion), bundled at `examples/data/`:

- `iris.csv` — Fisher's/Anderson's 150-flower Iris measurements (1936), the
  standard small classification benchmark.
- `mtcars.csv` — the 1974 *Motor Trend* US magazine road-test data for 32
  cars, a standard small regression benchmark (distributed with R's
  `datasets` package).

## Core conventions, briefly

- [`Dataset<Target>`](../src/core/dataset.rs) bundles a validated feature
  matrix with its targets. Construction fails fast on empty, ragged, or
  non-finite input, so a `Dataset` you're holding is already known-good.
- Every estimator follows the same shape: an unfitted config struct with
  `with_*` builders (each returning `Result<Self>` when the value needs
  validating), a `fit` method producing a `Fitted*` struct, and `predict`
  (classifiers also expose `predict_probabilities`/`decision_function`).
  Unsupervised transforms (`StandardScaler`, `PolynomialFeatures`, PCA,
  clustering) implement `Transform` instead of `Predict`.
- Every classifier orders classes deterministically (sorted by `Ord`), so
  probability columns and predictions are reproducible regardless of the
  order labels appeared in during training.
- Errors are a single structured `MlError` enum — no panics on bad input,
  no silent `NaN` propagation.

## Worked example 1: classifying Iris species

Full source: [`examples/iris_classification.rs`](../examples/iris_classification.rs)
— run it with `cargo run --example iris_classification --features csv`.

### Load the data

`dataset_from_csv_path` needs to know which column is the target; every
other column is parsed as an `f64` feature, in file order. `species` is the
last (index 4) of Iris's five columns:

```rust,ignore
let dataset: Dataset<String> = dataset_from_csv_path("examples/data/iris.csv", true, 4)?;
```

`Target` can be any `Clone + FromStr` type — `String` here, but an integer
or enum-backed label would work identically for a dataset that already
encodes classes numerically.

### Hold out a test split

```rust,ignore
let (train, test) = train_test_split(&dataset, SplitOptions::new(0.3).with_seed(0))?;
```

The split is seeded, so this is exactly reproducible. `train` and `test`
never overlap in either features or targets.

### Fit and evaluate

```rust,ignore
let model = RandomForestClassifier::new()
    .with_n_estimators(100)?
    .with_seed(0)
    .fit(&train)?;

let predictions = model.predict(test.records())?;
let accuracy = accuracy_score(test.targets(), predictions.view())?;
```

On this split, that scores **93.3% accuracy** — a `RandomForestClassifier`
with 100 trees, fit on 105 real flower measurements, correctly separating
three species it never saw at prediction time.

### Look past the single accuracy number

A single accuracy score hides *which* mistakes a model makes. `Iris` is a
good example of why that matters: `setosa` is trivially separable from the
other two, while `versicolor` and `virginica` overlap. The confusion matrix
and per-class report make that visible directly:

```rust,ignore
let matrix = confusion_matrix(test.targets(), predictions.view())?;
let report = classification_report(test.targets(), predictions.view())?;
```

Running the example prints exactly this asymmetry: `setosa` at perfect
precision and recall, `versicolor` and `virginica` occasionally confused
for each other. `RandomForestClassifier` also reports which features
mattered:

```rust,ignore
model.feature_importances()
// [sepal_length: 0.11, sepal_width: 0.02, petal_length: 0.39, petal_width: 0.48]
```

Petal measurements dominate — consistent with a century of botany showing
petal size is the more discriminating trait between these species, and a
useful sanity check that the model learned something real rather than
noise.

## Worked example 2: predicting fuel economy

Full source: [`examples/mtcars_regression.rs`](../examples/mtcars_regression.rs)
— run it with `cargo run --example mtcars_regression --features csv`.

### Load the data and split before scaling

`mpg` (miles per US gallon) is `mtcars.csv`'s first column:

```rust,ignore
let dataset: Dataset<f64> = dataset_from_csv_path("examples/data/mtcars.csv", true, 0)?;
let (train, test) = train_test_split(&dataset, SplitOptions::new(0.25).with_seed(0))?;
```

Splitting *before* scaling matters: fitting `StandardScaler` on the full
dataset would leak the test split's mean and variance into training,
quietly inflating the reported score. Fitting it on `train` alone and
applying that same transform to `test` keeps the test split genuinely
unseen:

```rust,ignore
let scaler = StandardScaler::default().fit(train.records())?;
let scaled_train = Dataset::new(scaler.transform(train.records())?, train.targets().to_owned())?;
let scaled_test = scaler.transform(test.records())?;
```

### Fit a regularized linear model

Ten correlated specs (cylinders, displacement, horsepower, weight, and so
on) describing 24 training cars is a natural fit for L2-regularized
regression rather than plain least squares:

```rust,ignore
let model = RidgeRegression::new(1.0)?.fit(&scaled_train)?;
let predictions = model.predict(scaled_test.view())?;

let r_squared = r2_score(test.targets(), predictions.view())?;
let mae = mean_absolute_error(test.targets(), predictions.view())?;
```

On this split: **R² of 0.87** and a **mean absolute error of 1.93 mpg** on
8 held-out cars — a real 10-feature-to-32-sample regression problem, not a
toy line-fitting exercise. The fitted coefficients put the largest
(negative) weight on `wt` (weight), matching the well-known real-world
relationship between a car's mass and its fuel economy.

## Where to go from here

Both examples above stop at a single train/test split. For anything you
plan to trust, cross-validate instead of relying on one split's luck:

- [`cross_validate`](../examples/cross_validation.rs) scores an estimator
  across several folds instead of one.
- [`grid_search`](../examples/grid_search.rs) and
  [`randomized_search`](../examples/randomized_search.rs) sweep
  hyperparameters (like `RidgeRegression`'s `alpha` above) under
  cross-validation instead of a guess.
- [`learning_curve`](../examples/learning_curve.rs) and
  [`validation_curve`](../examples/validation_curve.rs) diagnose whether a
  model needs more data or different regularization.
- [`variance_threshold`](../examples/variance_threshold.rs) and
  [`select_k_best`](../examples/select_k_best.rs) narrow down features
  before fitting, useful once a real dataset has more than a handful of
  candidate columns.
- [`pipeline`](../examples/pipeline.rs) chains preprocessing steps (like
  the `StandardScaler` above) into one fitted object instead of managing
  them by hand.

[`examples/README.md`](../examples/README.md) indexes every runnable
example in the crate, including the two used here, grouped by topic.
