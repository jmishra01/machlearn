//! Bagged and boosted decision-tree ensembles and their fitted models.

mod adaboost_classifier;
mod common;
mod gradient_boosting_classifier;
mod gradient_boosting_regressor;
mod random_forest_classifier;
mod random_forest_regressor;

pub use adaboost_classifier::{AdaBoostClassifier, FittedAdaBoostClassifier};
pub use common::MaxFeatures;
pub use gradient_boosting_classifier::{
    FittedGradientBoostingClassifier, GradientBoostingClassifier,
};
pub use gradient_boosting_regressor::{FittedGradientBoostingRegressor, GradientBoostingRegressor};
pub use random_forest_classifier::{FittedRandomForestClassifier, RandomForestClassifier};
pub use random_forest_regressor::{FittedRandomForestRegressor, RandomForestRegressor};
