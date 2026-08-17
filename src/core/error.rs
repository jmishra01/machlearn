use thiserror::Error;

/// Errors returned by `MachLearn` operations.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum MlError {
    /// A dataset has no rows.
    #[error("a dataset must contain at least one sample")]
    EmptySamples,

    /// A dataset has no feature columns.
    #[error("a dataset must contain at least one feature")]
    EmptyFeatures,

    /// A target or label collection has no entries.
    #[error("at least one target label is required")]
    EmptyTargets,

    /// Feature and target arrays contain different sample counts.
    #[error("feature rows ({feature_rows}) do not match target count ({target_count})")]
    MismatchedSampleCount {
        /// Number of rows in the feature matrix.
        feature_rows: usize,
        /// Number of entries in the target vector.
        target_count: usize,
    },

    /// An input has a different feature count from the fitted model.
    #[error("expected {expected} features, but received {actual}")]
    MismatchedFeatureCount {
        /// Feature count learned during fitting.
        expected: usize,
        /// Feature count supplied to the operation.
        actual: usize,
    },

    /// A feature contains NaN or infinity.
    #[error("feature at row {row}, column {column} is not finite")]
    NonFiniteFeature {
        /// Zero-based row containing the value.
        row: usize,
        /// Zero-based column containing the value.
        column: usize,
    },

    /// A feature contains positive or negative infinity.
    #[error("feature at row {row}, column {column} is infinite")]
    InfiniteFeature {
        /// Zero-based row containing the value.
        row: usize,
        /// Zero-based column containing the value.
        column: usize,
    },

    /// A fraction lies outside its supported interval.
    #[error("test fraction must be finite and strictly between 0 and 1; received {0}")]
    InvalidTestFraction(f64),

    /// A requested output range is empty, reversed, or non-finite.
    #[error(
        "feature range must contain finite values with minimum below maximum; received ({minimum}, {maximum})"
    )]
    InvalidFeatureRange {
        /// Requested lower bound.
        minimum: f64,
        /// Requested upper bound.
        maximum: f64,
    },

    /// A fitted label encoder did not observe a label during fitting.
    #[error("label at position {index} was not observed during fitting")]
    UnknownLabel {
        /// Position of the unknown label in the supplied input.
        index: usize,
    },

    /// An encoded class index lies outside the fitted class table.
    #[error(
        "encoded label at position {position} has class index {class_index}, but only {class_count} classes exist"
    )]
    InvalidClassIndex {
        /// Position of the invalid encoded label.
        position: usize,
        /// Invalid class index supplied by the caller.
        class_index: usize,
        /// Number of classes learned during fitting.
        class_count: usize,
    },

    /// A column contains no observed values from which to learn a statistic.
    #[error("feature column {column} contains only missing values")]
    AllValuesMissing {
        /// Zero-based feature column.
        column: usize,
    },

    /// A constant imputation value is NaN or infinite.
    #[error("imputation constant must be finite; received {0}")]
    InvalidImputationConstant(f64),

    /// Computing an imputation statistic produced a non-finite value.
    #[error("imputation statistic for feature column {column} is not finite")]
    NonFiniteImputationStatistic {
        /// Zero-based feature column.
        column: usize,
    },

    /// An operation requires more observations than were supplied.
    #[error("at least {required} samples are required, but only {actual} were supplied")]
    InsufficientSamples {
        /// Minimum supported sample count.
        required: usize,
        /// Actual sample count.
        actual: usize,
    },
}

/// Result type returned by `MachLearn` operations.
pub type Result<T> = std::result::Result<T, MlError>;
