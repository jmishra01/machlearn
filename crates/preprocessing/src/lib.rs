//! Data preparation: scaling, imputation, encoding, feature selection, and
//! dataset I/O.

/// Removing uninformative feature columns before fitting a model.
pub mod feature_selection;
/// Optional dataset input/output helpers.
pub mod io;
/// Data scaling and, later, feature-transformation utilities.
pub mod preprocessing;
