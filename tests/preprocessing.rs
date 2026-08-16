//! Integration tests for feature preprocessing.

use approx::assert_abs_diff_eq;
use machlearn::{MinMaxScaler, MlError, StandardScaler};
use ndarray::{Array2, Axis, array};

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
}

#[test]
fn standard_scaler_centers_and_scales_features() {
    let records = array![[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]];
    let scaler = StandardScaler::default().fit(records.view()).unwrap();
    let transformed = scaler.transform(records.view()).unwrap();

    assert_abs_diff_eq!(scaler.mean()[0], 3.0);
    assert_abs_diff_eq!(scaler.mean()[1], 2.0);
    assert_abs_diff_eq!(scaler.scale()[0], (8.0_f64 / 3.0).sqrt());
    assert_abs_diff_eq!(scaler.scale()[1], 1.0);

    for column in transformed.axis_iter(Axis(1)) {
        assert_abs_diff_eq!(column.iter().sum::<f64>(), 0.0, epsilon = 1.0e-12);
    }
    assert_matrix_close(
        &transformed.column(1).insert_axis(Axis(1)).to_owned(),
        &array![[0.0], [0.0], [0.0]],
    );
}

#[test]
fn standard_scaler_round_trips_data() {
    let records = array![[1.0, -5.0], [3.5, 0.0], [8.0, 10.0]];
    let scaler = StandardScaler::default().fit(records.view()).unwrap();
    let transformed = scaler.transform(records.view()).unwrap();
    let restored = scaler.inverse_transform(transformed.view()).unwrap();

    assert_matrix_close(&restored, &records);
}

#[test]
fn standard_scaler_can_disable_centering_and_scaling() {
    let records = array![[1.0, 2.0], [3.0, 4.0]];
    let scaler = StandardScaler::default()
        .with_mean(false)
        .with_std(false)
        .fit(records.view())
        .unwrap();

    assert_matrix_close(&scaler.transform(records.view()).unwrap(), &records);
}

#[test]
fn min_max_scaler_maps_to_default_range_and_handles_constants() {
    let records = array![[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]];
    let scaler = MinMaxScaler::default().fit(records.view()).unwrap();
    let transformed = scaler.transform(records.view()).unwrap();

    assert_matrix_close(&transformed, &array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]);
    assert_matrix_close(
        &scaler.inverse_transform(transformed.view()).unwrap(),
        &records,
    );
}

#[test]
fn min_max_scaler_supports_custom_ranges() {
    let records = array![[1.0], [3.0], [5.0]];
    let scaler = MinMaxScaler::new(-1.0, 1.0)
        .unwrap()
        .fit(records.view())
        .unwrap();

    assert_matrix_close(
        &scaler.transform(records.view()).unwrap(),
        &array![[-1.0], [0.0], [1.0]],
    );
}

#[test]
fn scalers_reject_wrong_feature_counts() {
    let fitted = StandardScaler::default()
        .fit(array![[1.0, 2.0]].view())
        .unwrap();
    let error = fitted.transform(array![[1.0]].view()).unwrap_err();

    assert_eq!(
        error,
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn min_max_scaler_rejects_invalid_ranges() {
    for (minimum, maximum) in [
        (1.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
    ] {
        assert!(matches!(
            MinMaxScaler::new(minimum, maximum),
            Err(MlError::InvalidFeatureRange { .. })
        ));
    }
}
