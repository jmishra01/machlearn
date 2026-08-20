# Dependency and license audit

Generated with `cargo license --all-features` (via `cargo-license` 0.7.0)
against the full dependency graph, including every optional feature
(`serde`, `parallel`, `csv`) so no platform- or feature-gated dependency is
missed.

## Result

Every dependency in the graph is under a permissive license compatible with
MachLearn's own `MIT OR Apache-2.0` license: MIT, Apache-2.0, BSD-2-Clause,
Zlib, BSL-1.0, Unicode-3.0, or Unlicense, several offered as an `OR` choice
between two or more of those. No copyleft license (GPL, LGPL, AGPL, MPL) is a
required option for any dependency — the one entry that lists LGPL-2.1
(`r-efi`, a Windows/UEFI-only transitive dependency) offers it only as one
option in an `Apache-2.0 OR LGPL-2.1-or-later OR MIT` choice, so it never
forces a copyleft obligation.

No action is required. Re-run the audit whenever a new direct or transitive
dependency is added, particularly before publishing a release:

```text
cargo install cargo-license   # one-time
cargo license --all-features
```

## Full license breakdown

| License | Count | Notable crates |
|---|---|---|
| Apache-2.0 OR MIT | 119 | `ndarray`, `rand`, `rayon`, `serde`, `criterion`, `proptest`, `thiserror` |
| MIT | 51 | `faer` and its `gemm`/`nano-gemm`/`pulp` numerical backend, `plotters` |
| MIT OR Unlicense | 8 | `csv`, `csv-core`, `memchr`, `aho-corasick`, `walkdir` |
| Apache-2.0 | 4 | `approx`, `ciborium` and its I/O crates |
| Apache-2.0 OR MIT OR Zlib | 2 | `bytemuck`, `bytemuck_derive` |
| Apache-2.0 OR BSD-2-Clause OR MIT | 2 | `zerocopy`, `zerocopy-derive` |
| Apache-2.0 OR LGPL-2.1-or-later OR MIT | 2 | `r-efi` (Windows/UEFI-only) |
| Apache-2.0 OR Apache-2.0 WITH LLVM-exception OR MIT | 4 | `rustix`, `linux-raw-sys`, `wasip2`, `wit-bindgen` |
| (Apache-2.0 OR MIT) AND Unicode-3.0 | 1 | `unicode-ident` |
| Apache-2.0 OR BSL-1.0 | 1 | `ryu` |
| BSD-2-Clause | 1 | `atomic-wait` |
