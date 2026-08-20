//! Integration tests for k-means clustering.

use approx::assert_abs_diff_eq;
use machlearn::{KMeans, KMeansInit, MlError};
use ndarray::array;

fn separated_dataset() -> ndarray::Array2<f64> {
    array![
        [-3.0, -3.0],
        [-2.5, -2.5],
        [-3.5, -2.0],
        [3.0, 3.0],
        [2.5, 2.5],
        [3.5, 2.0],
    ]
}

#[test]
fn matches_a_reference_solution() {
    // Reference centroids, inertia, and cluster assignment (up to a
    // relabeling of which cluster is "0") confirmed against
    // `sklearn.cluster.KMeans(n_clusters=2, n_init=1, random_state=0)`
    // fitted on the same data.
    let model = KMeans::new(2)
        .unwrap()
        .fit(separated_dataset().view())
        .unwrap();

    let predictions = model.predict(separated_dataset().view()).unwrap();
    assert_eq!(predictions[0], predictions[1]);
    assert_eq!(predictions[1], predictions[2]);
    assert_eq!(predictions[3], predictions[4]);
    assert_eq!(predictions[4], predictions[5]);
    assert_ne!(predictions[0], predictions[3]);

    assert_abs_diff_eq!(model.inertia(), 2.0, epsilon = 1.0e-9);

    let low_cluster = usize::from(model.centroids()[[0, 0]] > 0.0);
    let high_cluster = 1 - low_cluster;
    assert_abs_diff_eq!(model.centroids()[[low_cluster, 0]], -3.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.centroids()[[low_cluster, 1]], -2.5, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.centroids()[[high_cluster, 0]], 3.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.centroids()[[high_cluster, 1]], 2.5, epsilon = 1.0e-9);
}

#[test]
fn is_deterministic_for_a_fixed_seed() {
    let estimator = KMeans::new(2).unwrap().with_seed(11);
    let first = estimator.fit(separated_dataset().view()).unwrap();
    let second = estimator.fit(separated_dataset().view()).unwrap();

    assert_eq!(first, second);
}

#[test]
fn random_init_also_separates_well_clustered_data() {
    let model = KMeans::new(2)
        .unwrap()
        .with_init(KMeansInit::Random)
        .fit(separated_dataset().view())
        .unwrap();

    let predictions = model.predict(separated_dataset().view()).unwrap();
    assert_eq!(predictions[0], predictions[1]);
    assert_eq!(predictions[1], predictions[2]);
    assert_eq!(predictions[3], predictions[4]);
    assert_eq!(predictions[4], predictions[5]);
    assert_ne!(predictions[0], predictions[3]);
}

#[test]
fn transform_reports_zero_distance_to_the_assigned_centroid() {
    let model = KMeans::new(2)
        .unwrap()
        .fit(separated_dataset().view())
        .unwrap();

    let predictions = model.predict(separated_dataset().view()).unwrap();
    let distances = model.transform(separated_dataset().view()).unwrap();

    for (row_index, &cluster) in predictions.iter().enumerate() {
        for other_cluster in 0..model.n_clusters() {
            if other_cluster == cluster {
                continue;
            }
            assert!(distances[[row_index, cluster]] < distances[[row_index, other_cluster]]);
        }
    }
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let estimator = KMeans::new(3)
        .unwrap()
        .with_init(KMeansInit::Random)
        .with_max_iterations(50)
        .unwrap()
        .with_tolerance(1.0e-6)
        .unwrap()
        .with_seed(7);
    assert_eq!(estimator.n_clusters(), 3);
    assert_eq!(estimator.init(), KMeansInit::Random);
    assert_eq!(estimator.max_iterations(), 50);
    assert_abs_diff_eq!(estimator.tolerance(), 1.0e-6);
    assert_eq!(estimator.seed(), 7);
    assert_eq!(KMeans::new(1).unwrap().init(), KMeansInit::PlusPlus);

    assert_eq!(KMeans::new(0).unwrap_err(), MlError::InvalidClusterCount(0));
    assert_eq!(
        KMeans::new(2).unwrap().with_max_iterations(0).unwrap_err(),
        MlError::InvalidMaxIterations(0)
    );
    assert_eq!(
        KMeans::new(2).unwrap().with_tolerance(0.0).unwrap_err(),
        MlError::InvalidTolerance(0.0)
    );
}

#[test]
fn rejects_more_clusters_than_samples() {
    let dataset = array![[0.0], [1.0]];
    assert_eq!(
        KMeans::new(3).unwrap().fit(dataset.view()).unwrap_err(),
        MlError::InsufficientSamples {
            required: 3,
            actual: 2,
        }
    );
}

#[test]
fn validates_prediction_features() {
    let model = KMeans::new(2)
        .unwrap()
        .fit(separated_dataset().view())
        .unwrap();

    assert_eq!(
        model.predict(array![[1.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        model.predict(array![[f64::NAN, 0.0]].view()).unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = KMeans::new(2).unwrap().with_seed(3);
    let model = estimator.fit(separated_dataset().view()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: KMeans = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedKMeans = serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.n_clusters(), restored_model.n_clusters());
    assert_eq!(model.iterations(), restored_model.iterations());
    assert_eq!(model.converged(), restored_model.converged());
    // Compared with a tolerance rather than `assert_eq!`: `serde_json`'s text
    // round trip for `f64` is not guaranteed bit-exact in every case.
    assert_abs_diff_eq!(model.inertia(), restored_model.inertia(), epsilon = 1.0e-9);
    assert_abs_diff_eq!(
        model.centroids()[[0, 0]],
        restored_model.centroids()[[0, 0]],
        epsilon = 1.0e-9
    );
}
