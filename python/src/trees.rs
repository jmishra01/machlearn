use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::dataset::PyDataset;
use crate::error::to_py_err;

/// CART-style decision-tree regression.
#[pyclass(name = "DecisionTreeRegressor")]
pub struct PyDecisionTreeRegressor {
    config: machlearn::DecisionTreeRegressor,
    fitted: Option<machlearn::FittedDecisionTreeRegressor>,
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (max_depth=None))]
    fn new(max_depth: Option<usize>) -> Self {
        Self {
            config: machlearn::DecisionTreeRegressor::new().with_max_depth(max_depth),
            fitted: None,
        }
    }

    /// Fits the tree to `dataset`.
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

/// CART-style decision-tree classification.
///
/// `fit` takes raw `records`/integer `targets` arrays rather than a
/// `Dataset`, for the same reason as `LogisticRegression`:
/// `machlearn::DecisionTreeClassifier::fit` requires a label type
/// implementing `Ord`, which `f64` does not.
#[pyclass(name = "DecisionTreeClassifier")]
pub struct PyDecisionTreeClassifier {
    config: machlearn::DecisionTreeClassifier,
    fitted: Option<machlearn::FittedDecisionTreeClassifier<i64>>,
}

#[pymethods]
impl PyDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=None))]
    fn new(max_depth: Option<usize>) -> Self {
        Self {
            config: machlearn::DecisionTreeClassifier::new().with_max_depth(max_depth),
            fitted: None,
        }
    }

    /// Fits the tree to `records` and integer `targets`.
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

/// Bootstrap-aggregated random forest regression.
#[pyclass(name = "RandomForestRegressor")]
pub struct PyRandomForestRegressor {
    config: machlearn::RandomForestRegressor,
    fitted: Option<machlearn::FittedRandomForestRegressor>,
}

#[pymethods]
impl PyRandomForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::RandomForestRegressor::new()
                .with_n_estimators(n_estimators)
                .map_err(to_py_err)?
                .with_max_depth(max_depth),
            fitted: None,
        })
    }

    /// Fits the forest to `dataset`.
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

/// Bootstrap-aggregated random forest classification.
///
/// Same raw-array `fit` shape as `DecisionTreeClassifier`, for the same
/// `Ord`-bound reason.
#[pyclass(name = "RandomForestClassifier")]
pub struct PyRandomForestClassifier {
    config: machlearn::RandomForestClassifier,
    fitted: Option<machlearn::FittedRandomForestClassifier<i64>>,
}

#[pymethods]
impl PyRandomForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::RandomForestClassifier::new()
                .with_n_estimators(n_estimators)
                .map_err(to_py_err)?
                .with_max_depth(max_depth),
            fitted: None,
        })
    }

    /// Fits the forest to `records` and integer `targets`.
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
