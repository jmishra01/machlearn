//! Fundamental data structures, traits, and the internal numerical solver
//! shared by every `MachLearn` estimator crate.

/// Fundamental data structures and traits.
pub mod core;
/// Internal numerical solver backend (least-squares, ridge).
pub mod solver;
