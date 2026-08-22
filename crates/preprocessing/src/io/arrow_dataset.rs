use std::str::FromStr;

use arrow_array::{
    Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use ndarray::{Array1, Array2};

use machlearn_core::core::{MlError, Result};

/// Reads feature and target arrays from `batch`, mapping Arrow nulls to `NaN`.
///
/// One column, `target_column`, becomes the target; every other numeric
/// column becomes a dense `f64` feature column, in source order. Returns raw
/// arrays rather than a [`machlearn_core::core::Dataset`]: a null feature
/// cell becomes `NaN`, which `Dataset::new` rejects by design — impute (for
/// example with `SimpleImputer`) before constructing a `Dataset`, matching
/// this crate's missing-value policy.
///
/// # Errors
///
/// Returns an error when `target_column` is out of range, when a feature or
/// target column has an unsupported Arrow data type, or when the target
/// column contains a null cell (targets cannot be imputed generically).
pub fn arrays_from_record_batch<Target>(
    batch: &RecordBatch,
    target_column: usize,
) -> Result<(Array2<f64>, Array1<Target>)>
where
    Target: Clone + FromStr,
{
    let n_columns = batch.num_columns();
    if target_column >= n_columns {
        return Err(MlError::ArrowError(format!(
            "target column {target_column} is out of range for a batch with {n_columns} columns"
        )));
    }

    let n_rows = batch.num_rows();
    let n_features = n_columns.saturating_sub(1);
    let mut records = Array2::zeros((n_rows, n_features));

    let mut feature_column = 0;
    for column_index in 0..n_columns {
        if column_index == target_column {
            continue;
        }
        let array = batch.column(column_index).as_ref();
        for row in 0..n_rows {
            records[[row, feature_column]] = feature_value(array, row, column_index)?;
        }
        feature_column += 1;
    }

    let target_array = batch.column(target_column).as_ref();
    let mut targets = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        targets.push(target_value::<Target>(target_array, row, target_column)?);
    }

    Ok((records, Array1::from_vec(targets)))
}

/// Reads the `row`-th cell of a numeric Arrow `array` as `f64`, mapping a
/// null cell to `NaN`.
fn feature_value(array: &dyn Array, row: usize, column: usize) -> Result<f64> {
    if array.is_null(row) {
        return Ok(f64::NAN);
    }

    macro_rules! downcast_lossless {
        ($ty:ty) => {
            array
                .as_any()
                .downcast_ref::<$ty>()
                .map(|typed| f64::from(typed.value(row)))
        };
    }
    // `i64`/`u64` don't have a lossless `From<_> for f64` impl (an `f64`
    // mantissa is only 52 bits), but every other feature column in this
    // crate is already `f64`, so the same precision ceiling already applies
    // uniformly; this is an accepted, existing tradeoff, not a new one.
    macro_rules! downcast_lossy {
        ($ty:ty) => {
            array
                .as_any()
                .downcast_ref::<$ty>()
                .map(|typed| typed.value(row))
                .map(|value| {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        value as f64
                    }
                })
        };
    }

    downcast_lossless!(Float64Array)
        .or_else(|| downcast_lossless!(Float32Array))
        .or_else(|| downcast_lossy!(Int64Array))
        .or_else(|| downcast_lossless!(Int32Array))
        .or_else(|| downcast_lossless!(Int16Array))
        .or_else(|| downcast_lossless!(Int8Array))
        .or_else(|| downcast_lossy!(UInt64Array))
        .or_else(|| downcast_lossless!(UInt32Array))
        .or_else(|| downcast_lossless!(UInt16Array))
        .or_else(|| downcast_lossless!(UInt8Array))
        .ok_or_else(|| {
            MlError::ArrowError(format!(
                "feature column {column} has unsupported Arrow data type {:?}",
                array.data_type()
            ))
        })
}

/// Reads the `row`-th cell of an Arrow `array` as a target value, parsed
/// from its canonical string representation.
fn target_value<Target>(array: &dyn Array, row: usize, column: usize) -> Result<Target>
where
    Target: FromStr,
{
    if array.is_null(row) {
        return Err(MlError::ArrowError(format!(
            "target column {column} contains a null value at row {row}"
        )));
    }

    let text = match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|typed| typed.value(row).to_owned()),
        _ => None,
    };

    macro_rules! downcast_numeric_text {
        ($ty:ty) => {
            array
                .as_any()
                .downcast_ref::<$ty>()
                .map(|typed| typed.value(row).to_string())
        };
    }

    let text = text
        .or_else(|| downcast_numeric_text!(Float64Array))
        .or_else(|| downcast_numeric_text!(Float32Array))
        .or_else(|| downcast_numeric_text!(Int64Array))
        .or_else(|| downcast_numeric_text!(Int32Array))
        .or_else(|| downcast_numeric_text!(Int16Array))
        .or_else(|| downcast_numeric_text!(Int8Array))
        .or_else(|| downcast_numeric_text!(UInt64Array))
        .or_else(|| downcast_numeric_text!(UInt32Array))
        .or_else(|| downcast_numeric_text!(UInt16Array))
        .or_else(|| downcast_numeric_text!(UInt8Array))
        .ok_or_else(|| {
            MlError::ArrowError(format!(
                "target column {column} has unsupported Arrow data type {:?}",
                array.data_type()
            ))
        })?;

    text.parse::<Target>().map_err(|_untyped_error| {
        MlError::ArrowError(format!(
            "could not parse target {text:?} at row {row}, column {column}"
        ))
    })
}
