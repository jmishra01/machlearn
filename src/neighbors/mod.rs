//! Distance-based estimators and their fitted models.

mod common;
mod knn_classifier;
mod knn_regressor;

pub use common::Weighting;
pub use knn_classifier::{FittedKNeighborsClassifier, KNeighborsClassifier};
pub use knn_regressor::{FittedKNeighborsRegressor, KNeighborsRegressor};
