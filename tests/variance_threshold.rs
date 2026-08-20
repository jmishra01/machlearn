//! Integration tests for variance-threshold feature filtering.

use approx::assert_abs_diff_eq;
use machlearn::{MlError, VarianceThreshold};
use ndarray::array;

fn dataset() -> ndarray::Array2<f64> {
    array![
        [1.0, 2.0, 0.0, 5.0],
        [1.0, 4.0, 0.0, 3.0],
        [1.0, 6.0, 0.0, 1.0],
        [1.0, 8.0, 0.0, 8.0],
    ]
}

#[test]
fn matches_sklearn_default_threshold() {
    // Reference variances and selected columns confirmed against
    // `sklearn.feature_selection.VarianceThreshold(threshold=0.0)` fitted
    // on the same data.
    let fitted = VarianceThreshold::new().fit(dataset().view()).unwrap();

    assert_abs_diff_eq!(fitted.variances()[0], 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(fitted.variances()[1], 5.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(fitted.variances()[2], 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(fitted.variances()[3], 6.6875, epsilon = 1.0e-9);
    assert_eq!(fitted.selected_indices(), &[1, 3]);
    assert_eq!(fitted.n_selected_features(), 2);

    let transformed = fitted.transform(dataset().view()).unwrap();
    assert_eq!(
        transformed,
        array![[2.0, 5.0], [4.0, 3.0], [6.0, 1.0], [8.0, 8.0]]
    );
}

#[test]
fn a_higher_threshold_removes_more_features() {
    let fitted = VarianceThreshold::new()
        .with_threshold(6.0)
        .unwrap()
        .fit(dataset().view())
        .unwrap();

    assert_eq!(fitted.selected_indices(), &[3]);
    let transformed = fitted.transform(dataset().view()).unwrap();
    assert_eq!(transformed.ncols(), 1);
}

#[test]
fn every_feature_removed_produces_zero_columns() {
    let records = array![[1.0], [1.0], [1.0]];
    let fitted = VarianceThreshold::new().fit(records.view()).unwrap();

    assert_eq!(fitted.n_selected_features(), 0);
    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(transformed.shape(), &[3, 0]);
}

#[test]
fn exposes_configuration_and_validates_threshold() {
    let default = VarianceThreshold::default();
    assert_abs_diff_eq!(default.threshold(), 0.0);

    let configured = VarianceThreshold::new().with_threshold(0.5).unwrap();
    assert_abs_diff_eq!(configured.threshold(), 0.5);

    assert_eq!(
        VarianceThreshold::new().with_threshold(-1.0).unwrap_err(),
        MlError::InvalidVarianceThreshold(-1.0)
    );
    assert!(matches!(
        VarianceThreshold::new().with_threshold(f64::NAN),
        Err(MlError::InvalidVarianceThreshold(value)) if value.is_nan()
    ));
}

#[test]
fn validates_transform_features() {
    let fitted = VarianceThreshold::new().fit(dataset().view()).unwrap();

    assert_eq!(
        fitted.transform(array![[1.0, 2.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 4,
            actual: 2,
        }
    );
    assert_eq!(
        fitted
            .transform(array![[f64::NAN, 0.0, 0.0, 0.0]].view())
            .unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn fitted_filter_round_trips_through_serde() {
    let fitted = VarianceThreshold::new().fit(dataset().view()).unwrap();

    let json = serde_json::to_string(&fitted).unwrap();
    let restored: machlearn::FittedVarianceThreshold = serde_json::from_str(&json).unwrap();

    assert_eq!(fitted, restored);
}
