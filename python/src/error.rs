use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Converts a `machlearn` error into a Python exception.
///
/// A direct `impl From<MlError> for PyErr` isn't possible here: neither
/// `MlError` (defined in `machlearn-core`) nor `PyErr` (defined in `pyo3`)
/// is local to this crate, so Rust's orphan rule forbids it. Every fallible
/// call site instead does `.map_err(to_py_err)?`.
pub fn to_py_err(err: machlearn::MlError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
