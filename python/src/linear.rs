use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::dataset::PyDataset;
use crate::error::to_py_err;

/// Ordinary least-squares linear regression.
#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    config: machlearn::LinearRegression,
    fitted: Option<machlearn::FittedLinearRegression>,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            config: machlearn::LinearRegression::new().with_intercept(fit_intercept),
            fitted: None,
        }
    }

    /// Fits the model to `dataset`.
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

    /// Returns the fitted coefficients, one per input feature.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading coefficients"))?;
        Ok(fitted.coefficients().clone().into_pyarray(py))
    }

    /// Returns the fitted intercept.
    #[getter]
    fn intercept(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading intercept"))?;
        Ok(fitted.intercept())
    }
}

/// Binary logistic regression.
///
/// Takes raw `records`/`targets` arrays rather than a `Dataset`, because
/// `machlearn::LogisticRegression::fit` requires a label type implementing
/// `Ord` (to determine a deterministic negative/positive class order),
/// which `f64` (the label type `Dataset` is bound to in this crate) does
/// not implement — `NaN` breaks a total order. Integer class labels (`0`/`1`,
/// or any two distinct integers) sidestep that without introducing a second
/// public `Dataset` type in this starter phase.
#[pyclass(name = "LogisticRegression")]
pub struct PyLogisticRegression {
    config: machlearn::LogisticRegression,
    fitted: Option<machlearn::FittedLogisticRegression<i64>>,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            config: machlearn::LogisticRegression::new().with_intercept(fit_intercept),
            fitted: None,
        }
    }

    /// Fits the model to `records` and integer `targets` (exactly two
    /// distinct values).
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

    /// Predicts class labels for `records` using a 0.5 probability threshold.
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

    /// Predicts one probability column per class, in `[negative, positive]`
    /// order (see `classes`).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict_proba()"))?;
        let probabilities = fitted
            .predict_probabilities(records.as_array())
            .map_err(to_py_err)?;
        Ok(probabilities.into_pyarray(py))
    }

    /// Returns the two observed classes in `[negative, positive]` order.
    #[getter]
    fn classes(&self) -> PyResult<(i64, i64)> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading classes"))?;
        let classes = fitted.classes();
        Ok((classes[0], classes[1]))
    }

    /// Returns the fitted coefficients, one per input feature.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading coefficients"))?;
        Ok(fitted.coefficients().clone().into_pyarray(py))
    }

    /// Returns the fitted intercept.
    #[getter]
    fn intercept(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading intercept"))?;
        Ok(fitted.intercept())
    }
}

/// Ridge (L2-regularized) linear regression.
#[pyclass(name = "RidgeRegression")]
pub struct PyRidgeRegression {
    config: machlearn::RidgeRegression,
    fitted: Option<machlearn::FittedRidgeRegression>,
}

#[pymethods]
impl PyRidgeRegression {
    #[new]
    #[pyo3(signature = (alpha, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::RidgeRegression::new(alpha)
                .map_err(to_py_err)?
                .with_intercept(fit_intercept),
            fitted: None,
        })
    }

    /// Fits the model to `dataset`.
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

    /// Returns the fitted coefficients, one per input feature.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading coefficients"))?;
        Ok(fitted.coefficients().clone().into_pyarray(py))
    }

    /// Returns the fitted intercept.
    #[getter]
    fn intercept(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading intercept"))?;
        Ok(fitted.intercept())
    }
}

/// Lasso (L1-regularized) linear regression.
#[pyclass(name = "LassoRegression")]
pub struct PyLassoRegression {
    config: machlearn::LassoRegression,
    fitted: Option<machlearn::FittedLassoRegression>,
}

#[pymethods]
impl PyLassoRegression {
    #[new]
    #[pyo3(signature = (alpha, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::LassoRegression::new(alpha)
                .map_err(to_py_err)?
                .with_intercept(fit_intercept),
            fitted: None,
        })
    }

    /// Fits the model to `dataset`.
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

    /// Returns the fitted coefficients, one per input feature.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading coefficients"))?;
        Ok(fitted.coefficients().clone().into_pyarray(py))
    }

    /// Returns the fitted intercept.
    #[getter]
    fn intercept(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading intercept"))?;
        Ok(fitted.intercept())
    }
}

/// Elastic Net (combined L1/L2-regularized) linear regression.
#[pyclass(name = "ElasticNetRegression")]
pub struct PyElasticNetRegression {
    config: machlearn::ElasticNetRegression,
    fitted: Option<machlearn::FittedElasticNetRegression>,
}

#[pymethods]
impl PyElasticNetRegression {
    #[new]
    #[pyo3(signature = (alpha, l1_ratio, fit_intercept=true))]
    fn new(alpha: f64, l1_ratio: f64, fit_intercept: bool) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::ElasticNetRegression::new(alpha, l1_ratio)
                .map_err(to_py_err)?
                .with_intercept(fit_intercept),
            fitted: None,
        })
    }

    /// Fits the model to `dataset`.
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

    /// Returns the fitted coefficients, one per input feature.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading coefficients"))?;
        Ok(fitted.coefficients().clone().into_pyarray(py))
    }

    /// Returns the fitted intercept.
    #[getter]
    fn intercept(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading intercept"))?;
        Ok(fitted.intercept())
    }
}

/// Linear discriminant analysis (multi-class).
///
/// Takes raw `records`/integer `targets` arrays rather than a `Dataset`,
/// for the same reason as `LogisticRegression`: `fit` requires a label type
/// implementing `Ord`, which `f64` does not.
#[pyclass(name = "LinearDiscriminantAnalysis")]
pub struct PyLinearDiscriminantAnalysis {
    fitted: Option<machlearn::FittedLinearDiscriminantAnalysis<i64>>,
}

#[pymethods]
impl PyLinearDiscriminantAnalysis {
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// Fits the model to `records` and integer `targets`.
    fn fit(
        &mut self,
        records: PyReadonlyArray2<'_, f64>,
        targets: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let dataset =
            machlearn::Dataset::new(records.as_array().to_owned(), targets.as_array().to_owned())
                .map_err(to_py_err)?;
        self.fitted = Some(
            machlearn::LinearDiscriminantAnalysis::new()
                .fit(&dataset)
                .map_err(to_py_err)?,
        );
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

    /// Predicts one probability column per class, in `classes` order.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict_proba()"))?;
        let probabilities = fitted
            .predict_probabilities(records.as_array())
            .map_err(to_py_err)?;
        Ok(probabilities.into_pyarray(py))
    }

    /// Returns the observed classes in sorted order (matching the column
    /// order of `predict_proba`).
    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading classes"))?;
        let classes = ndarray::Array1::from_vec(fitted.classes().to_vec());
        Ok(classes.into_pyarray(py))
    }
}
