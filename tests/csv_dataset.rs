//! Integration tests for optional CSV dataset loading.
#![cfg(feature = "csv")]

use approx::assert_abs_diff_eq;
use machlearn::{MlError, dataset_from_csv_path, dataset_from_csv_reader};

#[test]
fn reads_a_dataset_with_a_header_row_and_trailing_target_column() {
    let csv = "feature_a,feature_b,label\n1.0,2.0,10.0\n3.0,4.0,20.0\n5.0,6.0,30.0\n";

    let dataset: machlearn::Dataset<f64> =
        dataset_from_csv_reader(csv.as_bytes(), true, 2).unwrap();

    assert_eq!(dataset.shape(), (3, 2));
    assert_abs_diff_eq!(dataset.records()[[0, 0]], 1.0);
    assert_abs_diff_eq!(dataset.records()[[0, 1]], 2.0);
    assert_abs_diff_eq!(dataset.targets()[0], 10.0);
    assert_abs_diff_eq!(dataset.targets()[2], 30.0);
}

#[test]
fn reads_a_dataset_with_a_leading_target_column_and_no_header() {
    let csv = "no,1.0,2.0\nyes,3.0,4.0\n";

    let dataset: machlearn::Dataset<String> =
        dataset_from_csv_reader(csv.as_bytes(), false, 0).unwrap();

    assert_eq!(dataset.shape(), (2, 2));
    assert_eq!(dataset.targets()[0], "no");
    assert_eq!(dataset.targets()[1], "yes");
    assert_abs_diff_eq!(dataset.records()[[1, 0]], 3.0);
}

#[test]
fn reads_a_dataset_from_a_file_path() {
    let mut path = std::env::temp_dir();
    path.push(format!("machlearn_csv_test_{}.csv", std::process::id()));
    std::fs::write(&path, "1.0,2.0,0\n3.0,4.0,1\n").unwrap();

    let dataset: machlearn::Dataset<u8> = dataset_from_csv_path(&path, false, 2).unwrap();

    std::fs::remove_file(&path).unwrap();

    assert_eq!(dataset.shape(), (2, 2));
    assert_eq!(dataset.targets()[0], 0);
    assert_eq!(dataset.targets()[1], 1);
}

#[test]
fn rejects_an_out_of_range_target_column() {
    let csv = "1.0,2.0\n";

    let result: Result<machlearn::Dataset<f64>, MlError> =
        dataset_from_csv_reader(csv.as_bytes(), false, 5);

    assert!(matches!(result, Err(MlError::CsvError(_))));
}

#[test]
fn rejects_an_unparsable_feature_value() {
    let csv = "not_a_number,2.0,0\n";

    let result: Result<machlearn::Dataset<u8>, MlError> =
        dataset_from_csv_reader(csv.as_bytes(), false, 2);

    assert!(matches!(result, Err(MlError::CsvError(_))));
}

#[test]
fn rejects_inconsistent_row_widths() {
    let csv = "1.0,2.0,0\n3.0,0\n";

    let result: Result<machlearn::Dataset<u8>, MlError> =
        dataset_from_csv_reader(csv.as_bytes(), false, 2);

    assert!(matches!(result, Err(MlError::CsvError(_))));
}

#[test]
fn propagates_dataset_validation_errors() {
    // Every column is the target, leaving no feature columns, which
    // `Dataset::new` rejects.
    let csv = "0\n1\n";

    let result: Result<machlearn::Dataset<u8>, MlError> =
        dataset_from_csv_reader(csv.as_bytes(), false, 0);

    assert_eq!(result.unwrap_err(), MlError::EmptyFeatures);
}
