# Interoperability evaluation

This document records the evaluation of three integration options tracked in
`TODO.md` section 8: Arrow/Polars interoperability, Python bindings via
PyO3, and WebAssembly support. Each is evaluated for scope, project-specific
fit, and a recommendation. None are implemented here; this is the "evaluate"
deliverable those checklist items ask for.

## WebAssembly support

**Verified, not just estimated.** `cargo check --target wasm32-unknown-unknown`
was run directly against this crate.

- With default features, compilation failed: `getrandom` (pulled in
  transitively by `rand`'s `sys_rng`/`thread_rng` default features) refuses
  to target `wasm32-unknown-unknown` without an explicit JS-interop opt-in.
  MachLearn never actually uses OS randomness — every RNG use in this crate
  is `ChaCha8Rng::seed_from_u64` with an explicit, deterministic seed (`KFold`,
  `StratifiedKFold`, `KMeans`, `RandomForestClassifier/Regressor`). Trimming
  `rand`'s features to `default-features = false, features = ["std",
  "std_rng"]` drops `getrandom` entirely and has already been applied to
  `Cargo.toml` (verified: full `cargo test --all-features` and strict Clippy
  still pass identically, confirming no behavior changed).
- With that fix, the crate compiles cleanly for `wasm32-unknown-unknown` with
  no default features, and with `serde` and `csv` both enabled.
- The `parallel` feature does **not** compile for `wasm32-unknown-unknown`:
  `rayon`'s dependency `atomic-wait` needs native platform threading
  primitives unavailable on bare wasm32 without additional tooling
  (`wasm-bindgen-rayon` plus cross-origin-isolation headers in the host
  page). This is expected and not worth chasing without a concrete need.

**Recommendation:** the library itself is already wasm-ready for
non-parallel use, at effectively zero cost (one dependency-feature trim,
already applied). Building a `wasm-bindgen` wrapper crate (JS-friendly types
across the boundary, a published npm package) is separate, additional work
and should wait for a concrete consumer. Until then, no further action is
needed beyond keeping the `rand` feature set minimal as new code is added.

## Python bindings with PyO3

**Not verified by compilation** — this is a structural/scope evaluation.

PyO3 bindings are conventionally a **separate crate** (a `cdylib`) in a
Cargo workspace, not code added directly to a library crate: `#[pyclass]`/
`#[pymethods]` macros generate CPython-specific glue that doesn't belong in
a pure-Rust library consumers may use without Python at all. A realistic
scope includes:

- A new workspace member (e.g. `machlearn-python`) depending on `machlearn`
  as a path dependency.
- `numpy` crate glue to convert between `PyArray2<f64>` and this crate's
  `ndarray` types — favorable, since MachLearn is `f64`-only dense `ndarray`
  throughout, which maps directly onto NumPy's default `float64` arrays with
  no type-dispatch complexity.
- `maturin` for building and publishing wheels, Python-side `.pyi` type
  stubs, and a second, independently versioned release artifact.
- CI changes: a second toolchain (Python + maturin) alongside the existing
  Rust-only pipeline.

**Recommendation: defer.** This is a new product surface, not an increment
to the library — multi-day effort, a second release cadence to maintain, and
binding churn risk while `machlearn`'s own API is still pre-1.0
(`publish = false`, no `0.1.0` release yet per `TODO.md` section 9). Revisit
once the estimator surface stabilizes. If pursued, structure it as a new
workspace member, never merge PyO3 code into `machlearn` itself.

## Arrow or Polars interoperability

**Not verified by compilation** — this is a structural/scope evaluation.

Motivation: let users already holding an Arrow `RecordBatch` or a Polars
`DataFrame` build a `Dataset` without a manual `ndarray` conversion step.

- Polars is itself built on Arrow (`DataFrame::to_arrow`/`Series` are
  backed by Arrow arrays), so an Arrow-first integration effectively covers
  both ecosystems through one code path — a bespoke Polars-specific
  integration would be redundant.
- MachLearn's existing missing-value policy (`NaN` is the only supported
  missing marker, documented in `README.md`) maps cleanly onto Arrow's
  nullable numeric columns: a null becomes `NaN` on conversion, reusing the
  crate's existing `SimpleImputer`/validation machinery rather than
  inventing a new null-handling policy.
- Conversion needs per-Arrow-type dispatch (`Int32Array`, `Float64Array`,
  etc. all need to funnel into a dense `f64` `Array2`), and `arrow-rs` is a
  comparatively heavy dependency tree to add behind a new optional feature.

**Recommendation: defer** until there is a concrete consumer need. When
pursued, target `arrow-rs` directly (not a Polars-specific path) behind a
new `arrow` feature flag, following the same optional-feature convention as
`serde`/`parallel`/`csv`, and map Arrow nulls to the crate's existing `NaN`
missing-value convention rather than introducing a second policy.
