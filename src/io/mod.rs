//! Optional dataset input/output helpers.

#[cfg(feature = "csv")]
mod csv_dataset;

#[cfg(feature = "csv")]
pub use csv_dataset::{dataset_from_csv_path, dataset_from_csv_reader};
