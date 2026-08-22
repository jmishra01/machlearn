use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Fraction of predicted labels that exactly match the actual labels.
#[pyfunction]
pub fn accuracy_score(
    actual: PyReadonlyArray1<'_, i64>,
    predicted: PyReadonlyArray1<'_, i64>,
) -> PyResult<f64> {
    machlearn::accuracy_score(actual.as_array(), predicted.as_array()).map_err(to_py_err)
}

/// Mean squared error between actual and predicted continuous targets.
#[pyfunction]
pub fn mean_squared_error(
    actual: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    machlearn::mean_squared_error(actual.as_array(), predicted.as_array()).map_err(to_py_err)
}

/// Coefficient of determination (R-squared).
#[pyfunction]
pub fn r2_score(
    actual: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    machlearn::r2_score(actual.as_array(), predicted.as_array()).map_err(to_py_err)
}

/// Precision: of the rows predicted positive, the fraction actually
/// positive (macro-averaged across classes for more than two classes).
#[pyfunction]
pub fn precision_score(
    actual: PyReadonlyArray1<'_, i64>,
    predicted: PyReadonlyArray1<'_, i64>,
) -> PyResult<f64> {
    machlearn::precision_score(actual.as_array(), predicted.as_array()).map_err(to_py_err)
}

/// Recall: of the rows actually positive, the fraction predicted positive
/// (macro-averaged across classes for more than two classes).
#[pyfunction]
pub fn recall_score(
    actual: PyReadonlyArray1<'_, i64>,
    predicted: PyReadonlyArray1<'_, i64>,
) -> PyResult<f64> {
    machlearn::recall_score(actual.as_array(), predicted.as_array()).map_err(to_py_err)
}

/// The harmonic mean of precision and recall (macro-averaged across
/// classes for more than two classes).
#[pyfunction]
pub fn f1_score(
    actual: PyReadonlyArray1<'_, i64>,
    predicted: PyReadonlyArray1<'_, i64>,
) -> PyResult<f64> {
    machlearn::f1_score(actual.as_array(), predicted.as_array()).map_err(to_py_err)
}

/// Area under the ROC curve for binary classification.
#[pyfunction]
pub fn roc_auc_score(
    actual: PyReadonlyArray1<'_, i64>,
    positive_probabilities: PyReadonlyArray1<'_, f64>,
    positive_label: i64,
) -> PyResult<f64> {
    machlearn::roc_auc_score(
        actual.as_array(),
        positive_probabilities.as_array(),
        &positive_label,
    )
    .map_err(to_py_err)
}

/// A confusion-matrix count grid paired with the class labels for its rows
/// and columns.
type ConfusionMatrixResult<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>);

/// Returns `(counts, classes)`: `counts[i, j]` is how many rows with actual
/// class `classes[i]` were predicted as class `classes[j]`.
#[pyfunction]
pub fn confusion_matrix<'py>(
    py: Python<'py>,
    actual: PyReadonlyArray1<'_, i64>,
    predicted: PyReadonlyArray1<'_, i64>,
) -> PyResult<ConfusionMatrixResult<'py>> {
    let matrix =
        machlearn::confusion_matrix(actual.as_array(), predicted.as_array()).map_err(to_py_err)?;
    let counts = matrix.counts().mapv(|count| count as i64);
    let classes = ndarray::Array1::from_vec(matrix.classes().to_vec());
    Ok((counts.into_pyarray(py), classes.into_pyarray(py)))
}
