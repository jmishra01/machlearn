//! Model-agnostic tools for inspecting a fitted estimator's behavior.

mod permutation_importance;

pub use permutation_importance::{PermutationImportance, permutation_importance};
