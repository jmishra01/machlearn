//! Loading a dataset from CSV data.

#[cfg(feature = "csv")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use machlearn::dataset_from_csv_reader;

    let csv = "sepal_length,sepal_width,species\n\
               5.1,3.5,setosa\n\
               6.4,3.2,versicolor\n\
               5.9,3.0,virginica\n";

    let dataset: machlearn::Dataset<String> = dataset_from_csv_reader(csv.as_bytes(), true, 2)?;

    println!("shape: {:?}", dataset.shape());
    println!("records: {:?}", dataset.records());
    println!("targets: {:?}", dataset.targets());
    Ok(())
}

#[cfg(not(feature = "csv"))]
fn main() {
    println!("Run with: cargo run --example csv_dataset --features csv");
}
