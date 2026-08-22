use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::error::to_py_err;

/// K-means clustering.
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    config: machlearn::KMeans,
    fitted: Option<machlearn::FittedKMeans>,
}

#[pymethods]
impl PyKMeans {
    #[new]
    fn new(n_clusters: usize) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::KMeans::new(n_clusters).map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Fits cluster centroids to `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Predicts the nearest cluster index for each row of `records`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict()"))?;
        let labels = fitted.predict(records.as_array()).map_err(to_py_err)?;
        let labels: ndarray::Array1<i64> = labels.mapv(|value| value as i64);
        Ok(labels.into_pyarray(py))
    }
}

/// Density-based spatial clustering (DBSCAN).
///
/// Unlike `KMeans`, DBSCAN has no `predict` for new points: it only labels
/// the rows it was fitted on (see `labels`). Noise points (reachable from no
/// core point) are encoded as `-1`, matching the common convention.
#[pyclass(name = "DBSCAN")]
pub struct PyDBSCAN {
    config: machlearn::DBSCAN,
    fitted: Option<machlearn::FittedDBSCAN>,
}

#[pymethods]
impl PyDBSCAN {
    #[new]
    fn new(eps: f64, min_samples: usize) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::DBSCAN::new(eps, min_samples).map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Fits DBSCAN to `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Returns one label per training row, in training order; `-1` marks
    /// noise.
    fn labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading labels"))?;
        let labels: ndarray::Array1<i64> = fitted
            .labels()
            .mapv(|label| label.map_or(-1, |cluster| cluster as i64));
        Ok(labels.into_pyarray(py))
    }

    /// Returns the number of clusters discovered (not counting noise).
    fn n_clusters(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before reading n_clusters"))?;
        Ok(fitted.n_clusters())
    }
}

/// Gaussian mixture model clustering (expectation-maximization).
#[pyclass(name = "GaussianMixture")]
pub struct PyGaussianMixture {
    config: machlearn::GaussianMixture,
    fitted: Option<machlearn::FittedGaussianMixture>,
}

#[pymethods]
impl PyGaussianMixture {
    #[new]
    fn new(n_components: usize) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::GaussianMixture::new(n_components).map_err(to_py_err)?,
            fitted: None,
        })
    }

    /// Fits the mixture to `records`.
    fn fit(&mut self, records: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.fitted = Some(self.config.fit(records.as_array()).map_err(to_py_err)?);
        Ok(())
    }

    /// Predicts the most likely component index for each row of `records`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict()"))?;
        let labels = fitted.predict(records.as_array()).map_err(to_py_err)?;
        let labels: ndarray::Array1<i64> = labels.mapv(|value| value as i64);
        Ok(labels.into_pyarray(py))
    }

    /// Predicts one membership-probability column per component.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        records: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("call fit() before predict_proba()"))?;
        let probabilities = fitted
            .predict_probabilities(records.as_array())
            .map_err(to_py_err)?;
        Ok(probabilities.into_pyarray(py))
    }
}
