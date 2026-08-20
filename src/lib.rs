#![doc = include_str!("../README.md")]
// `MachLearn` is a project name rather than an API symbol. Requiring code
// formatting for every occurrence makes the README less natural to read.
#![allow(clippy::doc_markdown)]

mod solver;

/// Clustering estimators.
pub mod cluster;
/// Fundamental data structures and traits.
pub mod core;
/// Dimensionality-reduction estimators.
pub mod decomposition;
/// Bagged decision-tree ensembles for classification and regression.
pub mod ensemble;
/// Linear estimators for regression and, later, classification.
pub mod linear_model;
/// Regression and, later, classification evaluation metrics.
pub mod metrics;
/// Dataset splitting and, later, model-selection utilities.
pub mod model_selection;
/// Probabilistic classifiers.
pub mod naive_bayes;
/// Distance-based estimators for classification and regression.
pub mod neighbors;
/// Data scaling and, later, feature-transformation utilities.
pub mod preprocessing;
/// Decision-tree estimators for classification and regression.
pub mod tree;

pub use crate::cluster::{FittedKMeans, KMeans, KMeansInit};
pub use crate::core::{Dataset, Fit, MlError, Predict, Result, Transform};
pub use crate::decomposition::{FittedPrincipalComponentAnalysis, PrincipalComponentAnalysis};
pub use crate::ensemble::{
    FittedRandomForestClassifier, FittedRandomForestRegressor, MaxFeatures, RandomForestClassifier,
    RandomForestRegressor,
};
pub use crate::linear_model::{
    ConvergenceReport, FittedLinearRegression, FittedLogisticRegression,
    FittedMulticlassLogisticRegression, FittedRidgeRegression, LinearRegression,
    LogisticRegression, MulticlassLogisticRegression, RidgeRegression,
};
pub use crate::metrics::{
    Averaging, ClassMetrics, ClassificationMetricOptions, ClassificationReport, ConfusionMatrix,
    ZeroDivision, accuracy_score, binary_log_loss, classification_report,
    classification_report_with_zero_division, confusion_matrix, f1_score, f1_score_with_options,
    mean_absolute_error, mean_squared_error, precision_score, precision_score_with_options,
    r2_score, recall_score, recall_score_with_options, roc_auc_score, root_mean_squared_error,
};
pub use crate::model_selection::{
    CrossValidationScores, Fold, GridSearchEntry, GridSearchResult, KFold, ParameterGrid,
    ParameterSet, ParameterValue, ScoreDirection, SplitOptions, StratifiedKFold, cross_validate,
    grid_search, train_test_split,
};
#[cfg(feature = "parallel")]
pub use crate::model_selection::{cross_validate_parallel, grid_search_parallel};
pub use crate::naive_bayes::{FittedGaussianNaiveBayes, GaussianNaiveBayes};
pub use crate::neighbors::{
    FittedKNeighborsClassifier, FittedKNeighborsRegressor, KNeighborsClassifier,
    KNeighborsRegressor, Weighting,
};
pub use crate::preprocessing::{
    FittedLabelEncoder, FittedMinMaxScaler, FittedPipeline, FittedSimpleImputer,
    FittedStandardScaler, FittedTransformer, ImputationStrategy, LabelEncoder, MinMaxScaler,
    Pipeline, SimpleImputer, StandardScaler, TransformerEstimator,
};
pub use crate::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, FittedDecisionTreeClassifier,
    FittedDecisionTreeRegressor,
};
