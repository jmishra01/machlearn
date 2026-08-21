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
/// Discriminant-analysis classifiers.
pub mod discriminant_analysis;
/// Bagged decision-tree ensembles for classification and regression.
pub mod ensemble;
/// Removing uninformative feature columns before fitting a model.
pub mod feature_selection;
/// Model-agnostic tools for inspecting a fitted estimator's behavior.
pub mod inspection;
/// Optional dataset input/output helpers.
pub mod io;
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

pub use crate::cluster::{
    DBSCAN, FittedDBSCAN, FittedGaussianMixture, FittedKMeans, GaussianMixture, KMeans, KMeansInit,
};
pub use crate::core::{Dataset, Fit, MlError, Predict, Result, Transform};
#[cfg(feature = "serde")]
pub use crate::core::{ENVELOPE_VERSION, ModelEnvelope};
pub use crate::decomposition::{FittedPrincipalComponentAnalysis, PrincipalComponentAnalysis};
pub use crate::discriminant_analysis::{
    FittedLinearDiscriminantAnalysis, LinearDiscriminantAnalysis,
};
pub use crate::ensemble::{
    AdaBoostClassifier, FittedAdaBoostClassifier, FittedGradientBoostingClassifier,
    FittedGradientBoostingRegressor, FittedRandomForestClassifier, FittedRandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, MaxFeatures, RandomForestClassifier,
    RandomForestRegressor,
};
pub use crate::feature_selection::{
    FittedSelectKBest, FittedVarianceThreshold, SelectKBest, VarianceThreshold, f_classif,
    f_regression,
};
pub use crate::inspection::{PermutationImportance, permutation_importance};
#[cfg(feature = "csv")]
pub use crate::io::{dataset_from_csv_path, dataset_from_csv_reader};
pub use crate::linear_model::{
    ConvergenceReport, ElasticNetRegression, FittedElasticNetRegression, FittedLassoRegression,
    FittedLinearRegression, FittedLogisticRegression, FittedMulticlassLogisticRegression,
    FittedRidgeRegression, LassoRegression, LinearRegression, LogisticRegression,
    MulticlassLogisticRegression, RidgeRegression,
};
pub use crate::metrics::{
    Averaging, ClassMetrics, ClassificationMetricOptions, ClassificationReport, ConfusionMatrix,
    ZeroDivision, accuracy_score, binary_log_loss, classification_report,
    classification_report_with_zero_division, confusion_matrix, f1_score, f1_score_with_options,
    mean_absolute_error, mean_squared_error, multiclass_log_loss, precision_score,
    precision_score_with_options, r2_score, recall_score, recall_score_with_options, roc_auc_score,
    roc_auc_score_ovr, root_mean_squared_error,
};
pub use crate::model_selection::{
    CrossValidationScores, CurveScores, Fold, GridSearchEntry, GridSearchResult, KFold,
    ParameterGrid, ParameterSet, ParameterValue, ScoreDirection, SplitOptions, StratifiedKFold,
    cross_validate, grid_search, learning_curve, randomized_search, train_test_split,
    validation_curve,
};
#[cfg(feature = "parallel")]
pub use crate::model_selection::{
    cross_validate_parallel, grid_search_parallel, randomized_search_parallel,
};
pub use crate::naive_bayes::{
    BernoulliNaiveBayes, FittedBernoulliNaiveBayes, FittedGaussianNaiveBayes,
    FittedMultinomialNaiveBayes, GaussianNaiveBayes, MultinomialNaiveBayes,
};
pub use crate::neighbors::{
    FittedKNeighborsClassifier, FittedKNeighborsRegressor, KNeighborsClassifier,
    KNeighborsRegressor, Weighting,
};
pub use crate::preprocessing::{
    FittedLabelEncoder, FittedMinMaxScaler, FittedOneHotEncoder, FittedPipeline,
    FittedPolynomialFeatures, FittedSimpleImputer, FittedStandardScaler, FittedTransformer,
    ImputationStrategy, LabelEncoder, MinMaxScaler, OneHotEncoder, Pipeline, PolynomialFeatures,
    SimpleImputer, StandardScaler, TransformerEstimator,
};
pub use crate::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, FittedDecisionTreeClassifier,
    FittedDecisionTreeRegressor,
};
