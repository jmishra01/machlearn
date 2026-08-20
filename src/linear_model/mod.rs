//! Linear estimators and their fitted models.

mod common;
mod convergence;
mod coordinate_descent;
mod elastic_net_regression;
mod lasso_regression;
mod linear_regression;
mod logistic_regression;
mod multiclass_logistic_regression;
mod ridge_regression;

pub use convergence::ConvergenceReport;
pub use elastic_net_regression::{ElasticNetRegression, FittedElasticNetRegression};
pub use lasso_regression::{FittedLassoRegression, LassoRegression};
pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use logistic_regression::{FittedLogisticRegression, LogisticRegression};
pub use multiclass_logistic_regression::{
    FittedMulticlassLogisticRegression, MulticlassLogisticRegression,
};
pub use ridge_regression::{FittedRidgeRegression, RidgeRegression};
