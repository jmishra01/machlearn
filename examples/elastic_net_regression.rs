//! Fitting Elastic Net regression and comparing it against Lasso.

use machlearn::{Dataset, ElasticNetRegression, LassoRegression, Result};
use ndarray::array;

fn main() -> Result<()> {
    let dataset = Dataset::new(
        array![
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 2.0],
            [5.0, 0.0, 0.0],
            [6.0, 1.0, 4.0],
        ],
        array![3.1, 4.8, 7.05, 9.0, 10.9, 13.15],
    )?;
    let query = array![[2.5, 0.5, 1.0]];

    let lasso = LassoRegression::new(0.5)?.fit(&dataset)?;
    let elastic_net = ElasticNetRegression::new(0.5, 0.3)?.fit(&dataset)?;

    println!("lasso coefficients: {:?}", lasso.coefficients());
    println!("elastic net coefficients: {:?}", elastic_net.coefficients());
    println!("lasso predict: {:?}", lasso.predict(query.view())?);
    println!(
        "elastic net predict: {:?}",
        elastic_net.predict(query.view())?
    );
    Ok(())
}
