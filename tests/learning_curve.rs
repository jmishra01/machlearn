//! Integration tests for the training-size learning curve.

#[cfg(feature = "serde")]
use approx::assert_abs_diff_eq;
use machlearn::{Dataset, KFold, LinearRegression, MlError, learning_curve, mean_squared_error};
use ndarray::array;

fn dataset() -> Dataset<f64> {
    Dataset::new(
        array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ],
        array![1.0, 2.9, 5.1, 7.0, 9.2, 11.0, 12.8, 15.1, 17.0, 19.2],
    )
    .unwrap()
}

#[test]
fn reports_one_row_per_training_size_and_one_column_per_fold() {
    let data = dataset();
    let folds = KFold::new(3).unwrap().split(data.n_samples()).unwrap();

    let scores = learning_curve(
        &LinearRegression::new(),
        &[3, 4, 5],
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert_eq!(scores.n_points(), 3);
    assert_eq!(scores.n_folds(), 3);
    assert_eq!(scores.train_scores().shape(), &[3, 3]);
    assert_eq!(scores.test_scores().shape(), &[3, 3]);
    assert!(scores.train_scores().iter().all(|value| *value >= 0.0));
    assert!(scores.test_scores().iter().all(|value| *value >= 0.0));
}

#[test]
fn a_well_fit_line_keeps_low_train_error_at_every_size() {
    let data = dataset();
    let folds = KFold::new(2).unwrap().split(data.n_samples()).unwrap();

    let scores = learning_curve(
        &LinearRegression::new(),
        &[3, 5],
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    for &value in &scores.train_scores_mean() {
        assert!(value < 0.1);
    }
}

#[test]
fn rejects_a_training_size_larger_than_a_folds_training_rows() {
    let data = dataset();
    let folds = KFold::new(2).unwrap().split(data.n_samples()).unwrap();
    let fold_train_size = folds[0].train_size();

    let error = learning_curve(
        &LinearRegression::new(),
        &[fold_train_size + 1],
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap_err();

    assert_eq!(
        error,
        MlError::InsufficientSamples {
            required: fold_train_size + 1,
            actual: fold_train_size,
        }
    );
}

#[test]
fn rejects_empty_training_sizes() {
    let data = dataset();
    let folds = KFold::new(2).unwrap().split(data.n_samples()).unwrap();

    let error = learning_curve(
        &LinearRegression::new(),
        &[],
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap_err();

    assert_eq!(error, MlError::EmptyCurvePoints);
}

#[test]
fn validates_folds() {
    let data = dataset();

    let error = learning_curve(
        &LinearRegression::new(),
        &[3],
        &data,
        &[],
        mean_squared_error,
    )
    .unwrap_err();

    assert_eq!(error, MlError::InvalidFoldCount { n_splits: 0 });
}

#[cfg(feature = "serde")]
#[test]
fn scores_round_trip_through_serde() {
    let data = dataset();
    let folds = KFold::new(2).unwrap().split(data.n_samples()).unwrap();
    let scores = learning_curve(
        &LinearRegression::new(),
        &[3, 5],
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    let json = serde_json::to_string(&scores).unwrap();
    let restored: machlearn::CurveScores = serde_json::from_str(&json).unwrap();

    // Compared with a tolerance rather than `assert_eq!`: `serde_json`'s text
    // round trip for `f64` is not guaranteed bit-exact in every case.
    assert_eq!(scores.n_points(), restored.n_points());
    assert_eq!(scores.n_folds(), restored.n_folds());
    for (left, right) in scores.train_scores().iter().zip(restored.train_scores()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
    for (left, right) in scores.test_scores().iter().zip(restored.test_scores()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
