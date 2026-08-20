use std::{fs::File, io::Read, path::Path, str::FromStr};

use ndarray::{Array1, Array2};

use crate::core::{Dataset, MlError, Result};

/// Reads a dense [`Dataset`] from CSV data.
///
/// One column, `target_column`, becomes the target; every other column is
/// parsed as an `f64` feature. Column order in the source data is preserved
/// for the remaining feature columns.
///
/// # Errors
///
/// Returns an error when a record cannot be read, when `target_column` is
/// out of range for a row, when a feature or target value cannot be parsed,
/// when rows have inconsistent feature-column counts, or when the resulting
/// dataset fails validation (see [`Dataset::new`]).
pub fn dataset_from_csv_reader<Target, R>(
    reader: R,
    has_headers: bool,
    target_column: usize,
) -> Result<Dataset<Target>>
where
    R: Read,
    Target: Clone + FromStr,
{
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(reader);

    let mut feature_rows: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Target> = Vec::new();
    let mut n_features = None;

    for (row_index, record) in csv_reader.records().enumerate() {
        let record = record.map_err(|error| MlError::CsvError(error.to_string()))?;
        if target_column >= record.len() {
            return Err(MlError::CsvError(format!(
                "target column {target_column} is out of range for row {row_index} with {} fields",
                record.len()
            )));
        }

        let mut row = Vec::with_capacity(record.len().saturating_sub(1));
        for (column_index, field) in record.iter().enumerate() {
            if column_index == target_column {
                let target = field.trim().parse::<Target>().map_err(|_untyped_error| {
                    MlError::CsvError(format!(
                        "could not parse target {field:?} at row {row_index}"
                    ))
                })?;
                targets.push(target);
            } else {
                let value: f64 = field.trim().parse().map_err(|_error| {
                    MlError::CsvError(format!(
                        "could not parse feature {field:?} at row {row_index}, column {column_index}"
                    ))
                })?;
                row.push(value);
            }
        }

        match n_features {
            None => n_features = Some(row.len()),
            Some(expected) if expected != row.len() => {
                return Err(MlError::CsvError(format!(
                    "row {row_index} has {} feature columns, but earlier rows have {expected}",
                    row.len()
                )));
            }
            Some(_) => {}
        }
        feature_rows.push(row);
    }

    let n_samples = feature_rows.len();
    let n_features = n_features.unwrap_or(0);
    let mut records = Array2::zeros((n_samples, n_features));
    for (row_index, row) in feature_rows.into_iter().enumerate() {
        for (column_index, value) in row.into_iter().enumerate() {
            records[[row_index, column_index]] = value;
        }
    }

    Dataset::new(records, Array1::from_vec(targets))
}

/// Reads a dense [`Dataset`] from a CSV file at `path`.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, or any error documented
/// on [`dataset_from_csv_reader`].
pub fn dataset_from_csv_path<Target, P>(
    path: P,
    has_headers: bool,
    target_column: usize,
) -> Result<Dataset<Target>>
where
    P: AsRef<Path>,
    Target: Clone + FromStr,
{
    let file = File::open(path).map_err(|error| MlError::CsvError(error.to_string()))?;
    dataset_from_csv_reader(file, has_headers, target_column)
}
