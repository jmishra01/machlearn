//! Integration tests for explicit missing-value handling.

use approx::assert_abs_diff_eq;
use machlearn::{MlError, SimpleImputer};
use ndarray::{Array2, array};

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn mean_strategy_learns_feature_statistics() {
    let records = array![[1.0, f64::NAN], [f64::NAN, 2.0], [3.0, 4.0]];
    let fitted = SimpleImputer::mean().fit(records.view()).unwrap();

    assert_abs_diff_eq!(fitted.fill_values()[0], 2.0);
    assert_abs_diff_eq!(fitted.fill_values()[1], 3.0);
    assert_matrix_close(
        &fitted.transform(records.view()).unwrap(),
        &array![[1.0, 3.0], [2.0, 2.0], [3.0, 4.0]],
    );
}

#[test]
fn median_strategy_handles_even_and_odd_observation_counts() {
    let records = array![
        [1.0, 10.0],
        [100.0, f64::NAN],
        [3.0, 30.0],
        [f64::NAN, 20.0]
    ];
    let fitted = SimpleImputer::median().fit(records.view()).unwrap();

    assert_abs_diff_eq!(fitted.fill_values()[0], 3.0);
    assert_abs_diff_eq!(fitted.fill_values()[1], 20.0);
}

#[test]
fn constant_strategy_handles_entirely_missing_columns() {
    let records = array![[f64::NAN, 1.0], [f64::NAN, f64::NAN]];
    let fitted = SimpleImputer::constant(-1.0)
        .unwrap()
        .fit(records.view())
        .unwrap();

    assert_matrix_close(
        &fitted.transform(records.view()).unwrap(),
        &array![[-1.0, 1.0], [-1.0, -1.0]],
    );
}

#[test]
fn statistical_strategies_reject_entirely_missing_columns() {
    let records = array![[f64::NAN, 1.0], [f64::NAN, 2.0]];
    assert_eq!(
        SimpleImputer::mean().fit(records.view()).unwrap_err(),
        MlError::AllValuesMissing { column: 0 }
    );
}

#[test]
fn rejects_infinite_values() {
    let records = array![[1.0, f64::INFINITY]];
    assert_eq!(
        SimpleImputer::mean().fit(records.view()).unwrap_err(),
        MlError::InfiniteFeature { row: 0, column: 1 }
    );
}

#[test]
fn rejects_non_finite_constants() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            SimpleImputer::constant(value),
            Err(MlError::InvalidImputationConstant(_))
        ));
    }
}

#[test]
fn rejects_wrong_feature_counts_during_transform() {
    let fitted = SimpleImputer::mean()
        .fit(array![[1.0, 2.0]].view())
        .unwrap();
    assert_eq!(
        fitted.transform(array![[f64::NAN]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
}

#[cfg(feature = "serde")]
#[test]
fn fitted_imputer_round_trips_through_serde() {
    let fitted = SimpleImputer::mean()
        .fit(array![[1.0, f64::NAN], [3.0, 2.0]].view())
        .unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let restored = serde_json::from_str(&json).unwrap();

    assert_eq!(fitted, restored);
}
