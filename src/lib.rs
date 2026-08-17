#![doc = include_str!("../README.md")]
// `MachLearn` is a project name rather than an API symbol. Requiring code
// formatting for every occurrence makes the README less natural to read.
#![allow(clippy::doc_markdown)]

/// Fundamental data structures and traits.
pub mod core;
/// Regression and, later, classification evaluation metrics.
pub mod metrics;
/// Dataset splitting and, later, model-selection utilities.
pub mod model_selection;
/// Data scaling and, later, feature-transformation utilities.
pub mod preprocessing;

pub use crate::core::{Dataset, Fit, MlError, Predict, Result, Transform};
pub use crate::metrics::{
    Averaging, ClassMetrics, ClassificationMetricOptions, ClassificationReport, ConfusionMatrix,
    ZeroDivision, accuracy_score, binary_log_loss, classification_report,
    classification_report_with_zero_division, confusion_matrix, f1_score, f1_score_with_options,
    mean_absolute_error, mean_squared_error, precision_score, precision_score_with_options,
    r2_score, recall_score, recall_score_with_options, roc_auc_score, root_mean_squared_error,
};
pub use crate::model_selection::{
    CrossValidationScores, Fold, KFold, ParameterGrid, ParameterSet, ParameterValue, SplitOptions,
    StratifiedKFold, cross_validate, train_test_split,
};
pub use crate::preprocessing::{
    FittedLabelEncoder, FittedMinMaxScaler, FittedPipeline, FittedSimpleImputer,
    FittedStandardScaler, FittedTransformer, ImputationStrategy, LabelEncoder, MinMaxScaler,
    Pipeline, SimpleImputer, StandardScaler, TransformerEstimator,
};
