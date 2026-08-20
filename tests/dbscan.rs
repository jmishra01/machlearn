//! Integration tests for DBSCAN density-based clustering.

use approx::assert_abs_diff_eq;
use machlearn::{DBSCAN, MlError};
use ndarray::array;

fn two_clusters_and_noise() -> ndarray::Array2<f64> {
    array![
        [1.0, 1.0],
        [1.2, 1.1],
        [0.8, 1.0],
        [5.0, 5.0],
        [5.1, 5.2],
        [5.2, 4.9],
        [25.0, 25.0],
        [1.0, 4.0],
    ]
}

#[test]
fn matches_sklearn_dbscan() {
    // Reference labels confirmed against
    // `sklearn.cluster.DBSCAN(eps=0.5, min_samples=3).fit_predict(X)` on
    // the same data: two dense clusters and two noise points.
    let model = DBSCAN::new(0.5, 3)
        .unwrap()
        .fit(two_clusters_and_noise().view())
        .unwrap();

    assert_eq!(model.n_clusters(), 2);
    assert_eq!(model.n_noise_points(), 2);

    let labels = model.labels();
    assert_eq!(labels[0], Some(0));
    assert_eq!(labels[1], Some(0));
    assert_eq!(labels[2], Some(0));
    assert_eq!(labels[3], Some(1));
    assert_eq!(labels[4], Some(1));
    assert_eq!(labels[5], Some(1));
    assert_eq!(labels[6], None);
    assert_eq!(labels[7], None);
}

#[test]
fn a_distance_exactly_at_eps_counts_as_a_neighbor() {
    // Reference confirmed against `sklearn.cluster.DBSCAN(eps=1.0,
    // min_samples=2)` on three colinear points one unit apart: the
    // boundary distance is included, chaining all three into one cluster.
    let records = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let model = DBSCAN::new(1.0, 2).unwrap().fit(records.view()).unwrap();

    assert_eq!(model.n_clusters(), 1);
    let labels = model.labels();
    assert_eq!(labels[0], Some(0));
    assert_eq!(labels[1], Some(0));
    assert_eq!(labels[2], Some(0));
}

#[test]
fn every_point_is_noise_when_min_samples_is_unreachable() {
    let records = two_clusters_and_noise();
    let model = DBSCAN::new(0.5, 100).unwrap().fit(records.view()).unwrap();

    assert_eq!(model.n_clusters(), 0);
    assert_eq!(model.n_noise_points(), records.nrows());
    assert!(model.labels().iter().all(Option::is_none));
}

#[test]
fn a_large_eps_merges_every_point_into_one_cluster() {
    let records = two_clusters_and_noise();
    let model = DBSCAN::new(1000.0, 2).unwrap().fit(records.view()).unwrap();

    assert_eq!(model.n_clusters(), 1);
    assert_eq!(model.n_noise_points(), 0);
    assert!(model.labels().iter().all(|label| *label == Some(0)));
}

#[test]
fn is_deterministic() {
    let records = two_clusters_and_noise();
    let estimator = DBSCAN::new(0.5, 3).unwrap();

    let first = estimator.fit(records.view()).unwrap();
    let second = estimator.fit(records.view()).unwrap();

    assert_eq!(first, second);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let estimator = DBSCAN::new(0.5, 3).unwrap();
    assert_abs_diff_eq!(estimator.eps(), 0.5);
    assert_eq!(estimator.min_samples(), 3);

    let updated = estimator
        .with_eps(1.5)
        .unwrap()
        .with_min_samples(5)
        .unwrap();
    assert_abs_diff_eq!(updated.eps(), 1.5);
    assert_eq!(updated.min_samples(), 5);

    assert_eq!(DBSCAN::new(0.0, 3).unwrap_err(), MlError::InvalidEps(0.0));
    assert_eq!(DBSCAN::new(-1.0, 3).unwrap_err(), MlError::InvalidEps(-1.0));
    assert!(matches!(
        DBSCAN::new(f64::NAN, 3),
        Err(MlError::InvalidEps(value)) if value.is_nan()
    ));
    assert_eq!(
        DBSCAN::new(0.5, 0).unwrap_err(),
        MlError::InvalidMinSamples(0)
    );
}

#[test]
fn validates_fit_features() {
    let estimator = DBSCAN::new(0.5, 3).unwrap();

    assert_eq!(
        estimator.fit(array![[f64::NAN]].view()).unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = DBSCAN::new(0.5, 3).unwrap();
    let model = estimator.fit(two_clusters_and_noise().view()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: DBSCAN = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedDBSCAN = serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model, restored_model);
}
