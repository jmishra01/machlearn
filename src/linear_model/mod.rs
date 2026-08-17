//! Linear estimators and their fitted models.

mod common;
mod linear_regression;
mod logistic_regression;
mod multiclass_logistic_regression;
mod ridge_regression;

pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use logistic_regression::{FittedLogisticRegression, LogisticRegression};
pub use multiclass_logistic_regression::{
    FittedMulticlassLogisticRegression, MulticlassLogisticRegression,
};
pub use ridge_regression::{FittedRidgeRegression, RidgeRegression};
