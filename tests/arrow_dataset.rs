//! Integration tests for optional Arrow `RecordBatch` dataset loading.
#![cfg(feature = "arrow")]

use std::sync::Arc;

use approx::assert_abs_diff_eq;
use arrow_array::{Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use machlearn::{Dataset, MlError, SimpleImputer, arrays_from_record_batch};

fn schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

#[test]
fn reads_features_and_a_trailing_target_column() {
    let batch = RecordBatch::try_new(
        schema(vec![
            Field::new("feature_a", DataType::Float64, false),
            Field::new("feature_b", DataType::Float64, false),
            Field::new("label", DataType::Float64, false),
        ]),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0])),
            Arc::new(Float64Array::from(vec![2.0, 4.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ],
    )
    .unwrap();

    let (records, targets) = arrays_from_record_batch::<f64>(&batch, 2).unwrap();

    assert_eq!(records.shape(), &[3, 2]);
    assert_abs_diff_eq!(records[[0, 0]], 1.0);
    assert_abs_diff_eq!(records[[0, 1]], 2.0);
    assert_abs_diff_eq!(targets[0], 10.0);
    assert_abs_diff_eq!(targets[2], 30.0);
}

#[test]
fn maps_null_feature_cells_to_nan() {
    let batch = RecordBatch::try_new(
        schema(vec![
            Field::new("feature_a", DataType::Float64, true),
            Field::new("label", DataType::Int32, false),
        ]),
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(5.0)])),
            Arc::new(Int32Array::from(vec![0, 1, 0])),
        ],
    )
    .unwrap();

    let (records, _targets) = arrays_from_record_batch::<i32>(&batch, 1).unwrap();

    assert_abs_diff_eq!(records[[0, 0]], 1.0);
    assert!(records[[1, 0]].is_nan());
    assert_abs_diff_eq!(records[[2, 0]], 5.0);
}

#[test]
fn imputing_null_features_composes_with_dataset_construction() {
    let batch = RecordBatch::try_new(
        schema(vec![
            Field::new("feature_a", DataType::Float64, true),
            Field::new("label", DataType::Float64, false),
        ]),
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(5.0)])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ],
    )
    .unwrap();

    let (raw_records, targets) = arrays_from_record_batch::<f64>(&batch, 1).unwrap();
    let imputer = SimpleImputer::mean().fit(raw_records.view()).unwrap();
    let clean_records = imputer.transform(raw_records.view()).unwrap();
    let dataset = Dataset::new(clean_records, targets).unwrap();

    assert_eq!(dataset.shape(), (3, 1));
    assert_abs_diff_eq!(dataset.records()[[1, 0]], 3.0);
}

#[test]
fn parses_string_targets_for_classification() {
    let batch = RecordBatch::try_new(
        schema(vec![
            Field::new("label", DataType::Utf8, false),
            Field::new("feature_a", DataType::Float64, false),
        ]),
        vec![
            Arc::new(StringArray::from(vec!["no", "yes"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ],
    )
    .unwrap();

    let (_records, targets) = arrays_from_record_batch::<String>(&batch, 0).unwrap();

    assert_eq!(targets[0], "no");
    assert_eq!(targets[1], "yes");
}

#[test]
fn rejects_an_out_of_range_target_column() {
    let batch = RecordBatch::try_new(
        schema(vec![Field::new("feature_a", DataType::Float64, false)]),
        vec![Arc::new(Float64Array::from(vec![1.0]))],
    )
    .unwrap();

    let result = arrays_from_record_batch::<f64>(&batch, 5);

    assert!(matches!(result, Err(MlError::ArrowError(_))));
}

#[test]
fn rejects_a_null_target_cell() {
    let batch = RecordBatch::try_new(
        schema(vec![
            Field::new("label", DataType::Float64, true),
            Field::new("feature_a", DataType::Float64, false),
        ]),
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), None])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ],
    )
    .unwrap();

    let result = arrays_from_record_batch::<f64>(&batch, 0);

    assert!(matches!(result, Err(MlError::ArrowError(_))));
}
