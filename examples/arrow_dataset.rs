//! Loading a dataset from an Arrow `RecordBatch`.

#[cfg(feature = "arrow")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use machlearn::{Dataset, SimpleImputer, arrays_from_record_batch};

    let schema = Arc::new(Schema::new(vec![
        Field::new("feature_a", DataType::Float64, false),
        Field::new("feature_b", DataType::Float64, true),
        Field::new("label", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 3.0, 5.0])),
            Arc::new(Float64Array::from(vec![Some(2.0), None, Some(4.0)])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ],
    )?;

    // A null cell becomes `NaN`, so impute before constructing a `Dataset`,
    // exactly as with any other raw `f64` array.
    let (raw_records, targets) = arrays_from_record_batch::<f64>(&batch, 2)?;
    let imputer = SimpleImputer::mean().fit(raw_records.view())?;
    let clean_records = imputer.transform(raw_records.view())?;
    let dataset = Dataset::new(clean_records, targets)?;

    println!("shape: {:?}", dataset.shape());
    println!("records: {:?}", dataset.records());
    println!("targets: {:?}", dataset.targets());
    Ok(())
}

#[cfg(not(feature = "arrow"))]
fn main() {
    println!("Run with: cargo run --example arrow_dataset --features arrow");
}
