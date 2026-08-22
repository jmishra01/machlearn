use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Gaussian Naive Bayes classification.
///
/// Supports any number of classes (unlike `LogisticRegression`, which is
/// binary-only). `fit` takes raw `records`/integer `targets` arrays rather
/// than a `Dataset`, for the same reason as `LogisticRegression`:
/// `machlearn::GaussianNaiveBayes::fit` requires a label type implementing
/// `Ord`, which `f64` does not.
#[pyclass(name = "GaussianNaiveBayes")]
pub struct PyGaussianNaiveBayes {
    config: machlearn::GaussianNaiveBayes,
    fitted: Option<machlearn::FittedGaussianNaiveBayes<i64>>,
}

#[pymethods]
impl PyGaussianNaiveBayes {
    #[new]
    fn new() -> Self {
        Self {
            config: machlearn::GaussianNaiveBayes::new(),
            fitted: None,
        }
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

/// Multinomial Naive Bayes classification (for count-like features, e.g.
/// word counts).
///
/// Same raw-array `fit` shape as `GaussianNaiveBayes`, for the same
/// `Ord`-bound reason.
#[pyclass(name = "MultinomialNaiveBayes")]
pub struct PyMultinomialNaiveBayes {
    config: machlearn::MultinomialNaiveBayes,
    fitted: Option<machlearn::FittedMultinomialNaiveBayes<i64>>,
}

#[pymethods]
impl PyMultinomialNaiveBayes {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_prior=true))]
    fn new(alpha: f64, fit_prior: bool) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::MultinomialNaiveBayes::default()
                .with_alpha(alpha)
                .map_err(to_py_err)?
                .with_fit_prior(fit_prior),
            fitted: None,
        })
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

/// Bernoulli Naive Bayes classification (for binary/binarized features).
///
/// Same raw-array `fit` shape as `GaussianNaiveBayes`, for the same
/// `Ord`-bound reason.
#[pyclass(name = "BernoulliNaiveBayes")]
pub struct PyBernoulliNaiveBayes {
    config: machlearn::BernoulliNaiveBayes,
    fitted: Option<machlearn::FittedBernoulliNaiveBayes<i64>>,
}

#[pymethods]
impl PyBernoulliNaiveBayes {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_prior=true, binarize=0.0))]
    fn new(alpha: f64, fit_prior: bool, binarize: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::BernoulliNaiveBayes::default()
                .with_alpha(alpha)
                .map_err(to_py_err)?
                .with_fit_prior(fit_prior)
                .with_binarize(binarize),
            fitted: None,
        })
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
