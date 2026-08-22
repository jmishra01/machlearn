/// Reports how an iterative solver stopped.
///
/// Every field describes the successful termination of the solver: fitting
/// returns an [`machlearn_core::core::MlError::OptimizationDidNotConverge`] error rather than
/// a report when the configured iteration budget is exhausted.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvergenceReport {
    pub(super) iterations: usize,
    pub(super) max_parameter_change: f64,
    pub(super) tolerance: f64,
}

impl ConvergenceReport {
    /// Returns the number of iterations completed before stopping.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the largest parameter change observed on the final iteration.
    ///
    /// This value is at most `tolerance` times the largest fitted parameter
    /// magnitude, which is the stopping condition that ended the fit.
    #[must_use]
    pub const fn max_parameter_change(&self) -> f64 {
        self.max_parameter_change
    }

    /// Returns the convergence tolerance used while fitting.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }
}
