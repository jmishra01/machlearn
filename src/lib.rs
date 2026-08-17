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
    mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
};
pub use crate::model_selection::{SplitOptions, train_test_split};
pub use crate::preprocessing::{
    FittedLabelEncoder, FittedMinMaxScaler, FittedPipeline, FittedSimpleImputer,
    FittedStandardScaler, FittedTransformer, ImputationStrategy, LabelEncoder, MinMaxScaler,
    Pipeline, SimpleImputer, StandardScaler, TransformerEstimator,
};
