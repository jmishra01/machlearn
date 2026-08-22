use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::dataset::PyDataset;
use crate::error::to_py_err;

/// Gradient-boosted decision-tree regression.
#[pyclass(name = "GradientBoostingRegressor")]
pub struct PyGradientBoostingRegressor {
    config: machlearn::GradientBoostingRegressor,
    fitted: Option<machlearn::FittedGradientBoostingRegressor>,
}

#[pymethods]
impl PyGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=Some(3)))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::GradientBoostingRegressor::new()
                .with_n_estimators(n_estimators)
                .map_err(to_py_err)?
                .with_learning_rate(learning_rate)
                .map_err(to_py_err)?
                .with_max_depth(max_depth),
            fitted: None,
        })
    }

    /// Fits the ensemble to `dataset`.
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

/// Gradient-boosted decision-tree classification (binary).
///
/// `fit` takes raw `records`/integer `targets` arrays rather than a
/// `Dataset`, for the same reason as `LogisticRegression`: it requires a
/// label type implementing `Ord`, which `f64` does not.
#[pyclass(name = "GradientBoostingClassifier")]
pub struct PyGradientBoostingClassifier {
    config: machlearn::GradientBoostingClassifier,
    fitted: Option<machlearn::FittedGradientBoostingClassifier<i64>>,
}

#[pymethods]
impl PyGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=Some(3)))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::GradientBoostingClassifier::new()
                .with_n_estimators(n_estimators)
                .map_err(to_py_err)?
                .with_learning_rate(learning_rate)
                .map_err(to_py_err)?
                .with_max_depth(max_depth),
            fitted: None,
        })
    }

    /// Fits the ensemble to `records` and integer `targets` (exactly two
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
}

/// `AdaBoost` classification (binary).
///
/// Same raw-array `fit` shape as `GradientBoostingClassifier`, for the same
/// `Ord`-bound reason.
#[pyclass(name = "AdaBoostClassifier")]
pub struct PyAdaBoostClassifier {
    config: machlearn::AdaBoostClassifier,
    fitted: Option<machlearn::FittedAdaBoostClassifier<i64>>,
}

#[pymethods]
impl PyAdaBoostClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0))]
    fn new(n_estimators: usize, learning_rate: f64) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::AdaBoostClassifier::new()
                .with_n_estimators(n_estimators)
                .map_err(to_py_err)?
                .with_learning_rate(learning_rate)
                .map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Fits the ensemble to `records` and integer `targets` (exactly two
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
}
