//! Ranking features by how much a fitted model's error grows when each is
//! independently shuffled.

use machlearn::{
    Dataset, RandomForestRegressor, Result, ScoreDirection, mean_squared_error,
    permutation_importance,
};
use ndarray::array;

fn main() -> Result<()> {
    // The target depends only on the first feature; the second is noise.
    let dataset = Dataset::new(
        array![
            [0.0, 9.0],
            [1.0, 2.0],
            [2.0, 7.0],
            [3.0, 1.0],
            [4.0, 8.0],
            [5.0, 3.0],
            [6.0, 6.0],
            [7.0, 4.0],
        ],
        array![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
    )?;
    let model = RandomForestRegressor::new()
        .with_n_estimators(50)?
        .fit(&dataset)?;

    let result = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        30,
        42,
    )?;

    println!("importances_mean: {:?}", result.importances_mean());
    println!("importances_std: {:?}", result.importances_std());
    Ok(())
}
