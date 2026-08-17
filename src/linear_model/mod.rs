//! Linear estimators and their fitted models.

mod common;
mod linear_regression;
mod ridge_regression;

pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use ridge_regression::{FittedRidgeRegression, RidgeRegression};
