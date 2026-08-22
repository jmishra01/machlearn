use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Feature-wise standardization (zero mean, unit variance).
#[pyclass(name = "StandardScaler")]
pub struct PyStandardScaler {
    config: machlearn::StandardScaler,
    fitted: Option<machlearn::FittedStandardScaler>,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            config: machlearn::StandardScaler::default()
                .with_mean(with_mean)
                .with_std(with_std),
            fitted: None,
        }
    }

    /// Learns the per-feature offset and scale from `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Applies the learned offset and scale to `records`.
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

    /// Fits to `records`, then immediately transforms them.
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
}

/// Replaces missing (`NaN`) feature values with a learned per-column
/// statistic.
#[pyclass(name = "SimpleImputer")]
pub struct PySimpleImputer {
    config: machlearn::SimpleImputer,
    fitted: Option<machlearn::FittedSimpleImputer>,
}

#[pymethods]
impl PySimpleImputer {
    /// Creates an imputer using the mean of each column's observed values.
    #[staticmethod]
    fn mean() -> Self {
        Self {
            config: machlearn::SimpleImputer::mean(),
            fitted: None,
        }
    }

    /// Creates an imputer using the median of each column's observed values.
    #[staticmethod]
    fn median() -> Self {
        Self {
            config: machlearn::SimpleImputer::median(),
            fitted: None,
        }
    }

    /// Creates an imputer that fills every missing value with `value`.
    #[staticmethod]
    fn constant(value: f64) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::SimpleImputer::constant(value).map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Learns the per-column fill values from `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Replaces missing values in `records` with the learned fill values.
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

    /// Fits to `records`, then immediately transforms them.
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

    /// Returns the learned per-column fill values.
    #[getter]
    fn fill_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading fill_values"))?;
        Ok(fitted.fill_values().clone().into_pyarray(py))
    }
}

/// Polynomial and interaction feature expansion.
#[pyclass(name = "PolynomialFeatures")]
pub struct PyPolynomialFeatures {
    config: machlearn::PolynomialFeatures,
    fitted: Option<machlearn::FittedPolynomialFeatures>,
}

#[pymethods]
impl PyPolynomialFeatures {
    #[new]
    #[pyo3(signature = (degree, include_bias=true, interaction_only=false))]
    fn new(degree: usize, include_bias: bool, interaction_only: bool) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::PolynomialFeatures::new(degree)
                .map_err(to_py_err)?
                .with_include_bias(include_bias)
                .with_interaction_only(interaction_only),
            fitted: None,
        })
    }

    /// Learns the output feature layout from `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Expands `records` into polynomial/interaction features.
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

    /// Fits to `records`, then immediately expands them.
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
}

/// Encodes string labels as sorted integer codes.
///
/// Takes a Python list of strings rather than a numpy array: labels are
/// categorical values (not features), and a `Vec<String>` converts
/// directly from a Python `list[str]` without needing numpy's less
/// ergonomic string-array support.
#[pyclass(name = "LabelEncoder")]
pub struct PyLabelEncoder {
    fitted: Option<machlearn::FittedLabelEncoder<String>>,
}

#[pymethods]
impl PyLabelEncoder {
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// Learns the sorted set of unique `labels`.
    fn fit(&mut self, labels: Vec<String>) -> PyResult<()> {
        let labels = ndarray::Array1::from_vec(labels);
        self.fitted = Some(
            machlearn::LabelEncoder::new()
                .fit(labels.view())
                .map_err(to_py_err)?,
        );
        Ok(())
    }

    /// Encodes `labels` as integer codes.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        labels: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before transform()"))?;
        let labels = ndarray::Array1::from_vec(labels);
        let encoded = fitted.transform(labels.view()).map_err(to_py_err)?;
        let encoded: ndarray::Array1<i64> = encoded.mapv(|value| value as i64);
        Ok(encoded.into_pyarray(py))
    }

    /// Decodes integer `codes` back into their original string labels.
    fn inverse_transform(&self, codes: Vec<i64>) -> PyResult<Vec<String>> {
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("call fit() before reading inverse_transform")
        })?;
        let codes: Vec<usize> = codes.iter().map(|&code| code as usize).collect();
        let codes = ndarray::Array1::from_vec(codes);
        let decoded = fitted.inverse_transform(codes.view()).map_err(to_py_err)?;
        Ok(decoded.to_vec())
    }

    /// Returns the observed classes in sorted order.
    #[getter]
    fn classes(&self) -> PyResult<Vec<String>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading classes"))?;
        Ok(fitted.classes().to_vec())
    }
}

/// Encodes string labels as one-hot indicator columns.
///
/// Same `Vec<String>` input convention as `LabelEncoder`, for the same
/// reason.
#[pyclass(name = "OneHotEncoder")]
pub struct PyOneHotEncoder {
    config: machlearn::OneHotEncoder,
    fitted: Option<machlearn::FittedOneHotEncoder<String>>,
}

#[pymethods]
impl PyOneHotEncoder {
    #[new]
    #[pyo3(signature = (drop_first=false))]
    fn new(drop_first: bool) -> Self {
        Self {
            config: machlearn::OneHotEncoder::new().with_drop_first(drop_first),
            fitted: None,
        }
    }

    /// Learns the sorted set of unique `labels`.
    fn fit(&mut self, labels: Vec<String>) -> PyResult<()> {
        let labels = ndarray::Array1::from_vec(labels);
        self.fitted = Some(self.config.fit(labels.view()).map_err(to_py_err)?);
        Ok(())
    }

    /// Encodes `labels` as one-hot indicator columns.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        labels: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before transform()"))?;
        let labels = ndarray::Array1::from_vec(labels);
        let encoded = fitted.transform(labels.view()).map_err(to_py_err)?;
        Ok(encoded.into_pyarray(py))
    }

    /// Returns the observed classes in sorted order (matching column order).
    #[getter]
    fn classes(&self) -> PyResult<Vec<String>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading classes"))?;
        Ok(fitted.classes().to_vec())
    }
}
