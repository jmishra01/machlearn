use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Principal component analysis.
#[pyclass(name = "PCA")]
pub struct PyPca {
    config: machlearn::PrincipalComponentAnalysis,
    fitted: Option<machlearn::FittedPrincipalComponentAnalysis>,
}

#[pymethods]
impl PyPca {
    #[new]
    #[pyo3(signature = (n_components=None))]
    fn new(n_components: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::PrincipalComponentAnalysis::new()
                .with_n_components(n_components)
                .map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Fits principal components to `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Projects `records` onto the fitted principal components.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before transform()"))?;
        let transformed = fitted.transform(records.as_array()).map_err(to_py_err)?;
        Ok(transformed.into_pyarray(py))
    }

    /// Fits to `records`, then immediately projects them.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self.config.fit(records.as_array()).map_err(to_py_err)?;
        let transformed = fitted.transform(records.as_array()).map_err(to_py_err)?;
        self.fitted = Some(fitted);
        Ok(transformed.into_pyarray(py))
    }

    /// Returns the fraction of total training-data variance captured by
    /// each retained component.
    #[getter]
    fn explained_variance_ratio<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("call fit() before reading explained_variance_ratio")
        })?;
        Ok(fitted.explained_variance_ratio().clone().into_pyarray(py))
    }
}
