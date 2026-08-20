//! Bagged decision-tree ensembles and their fitted models.

mod common;
mod random_forest_classifier;
mod random_forest_regressor;

pub use common::MaxFeatures;
pub use random_forest_classifier::{FittedRandomForestClassifier, RandomForestClassifier};
pub use random_forest_regressor::{FittedRandomForestRegressor, RandomForestRegressor};
