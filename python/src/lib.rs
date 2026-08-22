mod boosting;
mod cluster;
mod dataset;
mod decomposition;
mod error;
mod linear;
mod metrics;
mod model_selection;
mod naive_bayes;
mod neighbors;
mod preprocessing;
mod trees;

use pyo3::prelude::*;

#[pymodule]
fn machlearn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dataset::PyDataset>()?;
    m.add_class::<linear::PyLinearRegression>()?;
    m.add_class::<linear::PyLogisticRegression>()?;
    m.add_class::<linear::PyRidgeRegression>()?;
    m.add_class::<linear::PyLassoRegression>()?;
    m.add_class::<linear::PyElasticNetRegression>()?;
    m.add_class::<linear::PyLinearDiscriminantAnalysis>()?;
    m.add_class::<cluster::PyKMeans>()?;
    m.add_class::<cluster::PyDBSCAN>()?;
    m.add_class::<cluster::PyGaussianMixture>()?;
    m.add_class::<decomposition::PyPca>()?;
    m.add_class::<preprocessing::PyStandardScaler>()?;
    m.add_class::<preprocessing::PySimpleImputer>()?;
    m.add_class::<preprocessing::PyPolynomialFeatures>()?;
    m.add_class::<preprocessing::PyLabelEncoder>()?;
    m.add_class::<preprocessing::PyOneHotEncoder>()?;
    m.add_class::<trees::PyDecisionTreeRegressor>()?;
    m.add_class::<trees::PyDecisionTreeClassifier>()?;
    m.add_class::<trees::PyRandomForestRegressor>()?;
    m.add_class::<trees::PyRandomForestClassifier>()?;
    m.add_class::<boosting::PyGradientBoostingRegressor>()?;
    m.add_class::<boosting::PyGradientBoostingClassifier>()?;
    m.add_class::<boosting::PyAdaBoostClassifier>()?;
    m.add_class::<neighbors::PyKNeighborsRegressor>()?;
    m.add_class::<neighbors::PyKNeighborsClassifier>()?;
    m.add_class::<naive_bayes::PyGaussianNaiveBayes>()?;
    m.add_class::<naive_bayes::PyMultinomialNaiveBayes>()?;
    m.add_class::<naive_bayes::PyBernoulliNaiveBayes>()?;
    m.add_class::<model_selection::PyKFold>()?;
    m.add_class::<model_selection::PyStratifiedKFold>()?;
    m.add_function(wrap_pyfunction!(model_selection::train_test_split, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::recall_score, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::roc_auc_score, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::confusion_matrix, m)?)?;
    Ok(())
}
