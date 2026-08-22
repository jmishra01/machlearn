use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::dataset::PyDataset;
use crate::error::to_py_err;

/// Distance-weighted k-nearest-neighbors regression.
#[pyclass(name = "KNeighborsRegressor")]
pub struct PyKNeighborsRegressor {
    config: machlearn::KNeighborsRegressor,
    fitted: Option<machlearn::FittedKNeighborsRegressor>,
}

#[pymethods]
impl PyKNeighborsRegressor {
    #[new]
    fn new(n_neighbors: usize) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::KNeighborsRegressor::new(n_neighbors).map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Stores `dataset` for later neighbor lookups.
    fn fit(&mut self, dataset: &PyDataset) -> PyResult<()> {
        self.fitted = Some(self.config.fit(&dataset.inner).map_err(to_py_err)?);
        Ok(())
    }

    /// Predicts continuous targets for `records`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict()"))?;
        let predictions = fitted.predict(records.as_array()).map_err(to_py_err)?;
        Ok(predictions.into_pyarray(py))
    }
}

/// Distance-weighted k-nearest-neighbors classification.
///
/// `fit` takes raw `records`/integer `targets` arrays rather than a
/// `Dataset`, for the same reason as `LogisticRegression`:
/// `machlearn::KNeighborsClassifier::fit` requires a label type implementing
/// `Ord`, which `f64` does not.
#[pyclass(name = "KNeighborsClassifier")]
pub struct PyKNeighborsClassifier {
    config: machlearn::KNeighborsClassifier,
    fitted: Option<machlearn::FittedKNeighborsClassifier<i64>>,
}

#[pymethods]
impl PyKNeighborsClassifier {
    #[new]
    fn new(n_neighbors: usize) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::KNeighborsClassifier::new(n_neighbors).map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Stores `records` and integer `targets` for later neighbor lookups.
    fn fit(
        &mut self,
        records: PyReadonlyArray2<'_, f64>,
        targets: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let dataset =
            machlearn::Dataset::new(records.as_array().to_owned(), targets.as_array().to_owned())
                .map_err(to_py_err)?;
        self.fitted = Some(self.config.fit(&dataset).map_err(to_py_err)?);
        Ok(())
    }

    /// Predicts class labels for `records`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict()"))?;
        let predictions = fitted.predict(records.as_array()).map_err(to_py_err)?;
        Ok(predictions.into_pyarray(py))
    }
}
