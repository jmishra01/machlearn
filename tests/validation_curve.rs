//! Integration tests for the hyperparameter validation curve.

#[cfg(feature = "serde")]
use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, KFold, MlError, ParameterValue, RidgeRegression, mean_squared_error, validation_curve,
};
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

fn alphas() -> [ParameterValue; 3] {
    [
        ParameterValue::from(0.01),
        ParameterValue::from(1.0),
        ParameterValue::from(100.0),
    ]
}

#[test]
fn reports_one_row_per_parameter_value_and_one_column_per_fold() {
    let data = dataset();
    let folds = KFold::new(3).unwrap().split(data.n_samples()).unwrap();

    let scores = validation_curve(
        &alphas(),
        |value| RidgeRegression::new(value.as_f64().unwrap()),
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert_eq!(scores.n_points(), 3);
    assert_eq!(scores.n_folds(), 3);
    assert_eq!(scores.train_scores().shape(), &[3, 3]);
    assert_eq!(scores.test_scores().shape(), &[3, 3]);
}

#[test]
fn heavier_regularization_increases_error_on_a_linear_trend() {
    let data = dataset();
    let folds = KFold::new(3).unwrap().split(data.n_samples()).unwrap();

    let scores = validation_curve(
        &alphas(),
        |value| RidgeRegression::new(value.as_f64().unwrap()),
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    let train_mean = scores.train_scores_mean();
    let test_mean = scores.test_scores_mean();
    assert!(train_mean[0] < train_mean[2]);
    assert!(test_mean[0] < test_mean[2]);
}

#[test]
fn rejects_an_empty_parameter_range() {
    let data = dataset();
    let folds = KFold::new(2).unwrap().split(data.n_samples()).unwrap();

    let error = validation_curve(
        &[] as &[ParameterValue],
        |value| RidgeRegression::new(value.as_f64().unwrap()),
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap_err();

    assert_eq!(error, MlError::EmptyCurvePoints);
}

#[test]
fn propagates_factory_errors() {
    let data = dataset();
    let folds = KFold::new(2).unwrap().split(data.n_samples()).unwrap();

    let error = validation_curve(
        &[ParameterValue::from(-1.0)],
        |value| RidgeRegression::new(value.as_f64().unwrap()),
        &data,
        &folds,
        mean_squared_error,
    )
    .unwrap_err();

    assert_eq!(error, MlError::InvalidRegularization(-1.0));
}

#[test]
fn validates_folds() {
    let data = dataset();

    let error = validation_curve(
        &alphas(),
        |value| RidgeRegression::new(value.as_f64().unwrap()),
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
    let scores = validation_curve(
        &alphas(),
        |value| RidgeRegression::new(value.as_f64().unwrap()),
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
