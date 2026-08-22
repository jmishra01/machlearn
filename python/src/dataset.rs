use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// A validated dense feature matrix paired with one target per sample.
#[pyclass(name = "Dataset")]
pub struct PyDataset {
    pub(crate) inner: machlearn::Dataset<f64>,
}

#[pymethods]
impl PyDataset {
    #[new]
    fn new(
        records: PyReadonlyArray2<'_, f64>,
        targets: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let inner =
            machlearn::Dataset::new(records.as_array().to_owned(), targets.as_array().to_owned())
                .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Returns the `(n_samples, n_features)` shape.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Returns the feature matrix as a numpy array.
    #[getter]
    fn records<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.records().to_owned().into_pyarray(py)
    }

    /// Returns the target vector as a numpy array.
    #[getter]
    fn targets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.targets().to_owned().into_pyarray(py)
    }
}
