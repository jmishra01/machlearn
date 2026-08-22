//! Optional dataset input/output helpers.

#[cfg(feature = "arrow")]
mod arrow_dataset;
#[cfg(feature = "csv")]
mod csv_dataset;

#[cfg(feature = "arrow")]
pub use arrow_dataset::arrays_from_record_batch;
#[cfg(feature = "csv")]
pub use csv_dataset::{dataset_from_csv_path, dataset_from_csv_reader};
