//! Evaluating continuous predictions with regression metrics.

use machlearn::{
    Result, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
};
use ndarray::array;

fn main() -> Result<()> {
    let actual = array![3.0, -0.5, 2.0, 7.0];
    let predicted = array![2.5, 0.0, 2.0, 8.0];

    println!(
        "MSE:  {:.6}",
        mean_squared_error(actual.view(), predicted.view())?
    );
    println!(
        "RMSE: {:.6}",
        root_mean_squared_error(actual.view(), predicted.view())?
    );
    println!(
        "MAE:  {:.6}",
        mean_absolute_error(actual.view(), predicted.view())?
    );
    println!("R²:   {:.6}", r2_score(actual.view(), predicted.view())?);
    Ok(())
}
