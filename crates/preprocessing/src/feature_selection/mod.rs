//! Feature selection: removing uninformative feature columns before fitting
//! a model.

mod select_k_best;
mod univariate;
mod variance_threshold;

pub use select_k_best::{FittedSelectKBest, SelectKBest};
pub use univariate::{f_classif, f_regression};
pub use variance_threshold::{FittedVarianceThreshold, VarianceThreshold};
