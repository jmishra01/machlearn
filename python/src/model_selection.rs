use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::dataset::PyDataset;
use crate::error::to_py_err;

/// Splits `dataset` into a train/test pair.
#[pyfunction]
#[pyo3(signature = (dataset, test_fraction, seed=42, shuffle=true))]
pub fn train_test_split(
    dataset: &PyDataset,
    test_fraction: f64,
    seed: u64,
    shuffle: bool,
) -> PyResult<(PyDataset, PyDataset)> {
    let options = machlearn::SplitOptions::new(test_fraction)
        .with_seed(seed)
        .with_shuffle(shuffle);
    let (train, test) = machlearn::train_test_split(&dataset.inner, options).map_err(to_py_err)?;
    Ok((PyDataset { inner: train }, PyDataset { inner: test }))
}

/// One fold's train/test index arrays.
type FoldIndices<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

fn folds_to_py(py: Python<'_>, folds: Vec<machlearn::Fold>) -> Vec<FoldIndices<'_>> {
    folds
        .into_iter()
        .map(|fold| {
            let train: ndarray::Array1<i64> = fold
                .train_indices()
                .iter()
                .map(|&index| index as i64)
                .collect();
            let test: ndarray::Array1<i64> = fold
                .test_indices()
                .iter()
                .map(|&index| index as i64)
                .collect();
            (train.into_pyarray(py), test.into_pyarray(py))
        })
        .collect()
}

/// Ordered K-fold cross-validation index splitting.
#[pyclass(name = "KFold")]
pub struct PyKFold {
    config: machlearn::KFold,
}

#[pymethods]
impl PyKFold {
    #[new]
    #[pyo3(signature = (n_splits, shuffle=false, seed=42))]
    fn new(n_splits: usize, shuffle: bool, seed: u64) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::KFold::new(n_splits)
                .map_err(to_py_err)?
                .with_shuffle(shuffle)
                .with_seed(seed),
        })
    }

    /// Returns `[(train_indices, test_indices), ...]`, one pair per fold,
    /// for a dataset with `n_samples` rows.
    fn split<'py>(&self, py: Python<'py>, n_samples: usize) -> PyResult<Vec<FoldIndices<'py>>> {
        let folds = self.config.split(n_samples).map_err(to_py_err)?;
        Ok(folds_to_py(py, folds))
    }
}

/// Stratified K-fold cross-validation index splitting, preserving each
/// fold's class proportions.
#[pyclass(name = "StratifiedKFold")]
pub struct PyStratifiedKFold {
    config: machlearn::StratifiedKFold,
}

#[pymethods]
impl PyStratifiedKFold {
    #[new]
    #[pyo3(signature = (n_splits, shuffle=false, seed=42))]
    fn new(n_splits: usize, shuffle: bool, seed: u64) -> PyResult<Self> {
        Ok(Self {
            config: machlearn::StratifiedKFold::new(n_splits)
                .map_err(to_py_err)?
                .with_shuffle(shuffle)
                .with_seed(seed),
        })
    }

    /// Returns `[(train_indices, test_indices), ...]`, one pair per fold,
    /// for integer class `targets`.
    fn split<'py>(&self, py: Python<'py>, targets: Vec<i64>) -> PyResult<Vec<FoldIndices<'py>>> {
        let folds = self.config.split(&targets).map_err(to_py_err)?;
        Ok(folds_to_py(py, folds))
    }
}
