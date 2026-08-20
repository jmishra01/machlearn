//! Discovering density-connected clusters and marking outliers as noise.

use machlearn::{DBSCAN, Result};
use ndarray::array;

fn main() -> Result<()> {
    let records = array![
        [1.0, 1.0],
        [1.2, 1.1],
        [0.8, 1.0],
        [5.0, 5.0],
        [5.1, 5.2],
        [5.2, 4.9],
        [25.0, 25.0],
        [1.0, 4.0],
    ];
    let model = DBSCAN::new(0.5, 3)?.fit(records.view())?;

    println!("labels: {:?}", model.labels());
    println!("n_clusters: {}", model.n_clusters());
    println!("n_noise_points: {}", model.n_noise_points());
    Ok(())
}
