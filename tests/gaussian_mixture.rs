//! Integration tests for Gaussian mixture models fit by
//! expectation-maximization.

use approx::assert_abs_diff_eq;
use machlearn::{GaussianMixture, MlError};
use ndarray::array;

fn separated_dataset() -> ndarray::Array2<f64> {
    array![
        [-3.0, -3.0],
        [-2.5, -2.5],
        [-3.5, -2.0],
        [-2.8, -3.2],
        [-3.2, -2.8],
        [3.0, 3.0],
        [2.5, 2.5],
        [3.5, 2.0],
        [2.8, 3.2],
        [3.2, 2.8],
    ]
}

#[test]
fn matches_a_reference_solution() {
    // Reference weights, means, covariances, convergence, iteration count,
    // and log-likelihood (up to a relabeling of which component is "0")
    // confirmed against `sklearn.mixture.GaussianMixture(n_components=2,
    // n_init=1, init_params="kmeans", random_state=0)` fitted on the same
    // data.
    let model = GaussianMixture::new(2)
        .unwrap()
        .fit(separated_dataset().view())
        .unwrap();

    assert!(model.converged());
    assert_eq!(model.n_iterations(), 2);
    assert_abs_diff_eq!(
        model.log_likelihood(),
        -1.465_312_461_789_04,
        epsilon = 1.0e-9
    );

    assert_abs_diff_eq!(model.weights()[0], 0.5, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.weights()[1], 0.5, epsilon = 1.0e-9);

    let low_component = usize::from(model.means()[[0, 0]] > 0.0);
    let high_component = 1 - low_component;
    assert_abs_diff_eq!(model.means()[[low_component, 0]], -3.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.means()[[low_component, 1]], -2.7, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.means()[[high_component, 0]], 3.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(model.means()[[high_component, 1]], 2.7, epsilon = 1.0e-9);

    for component in [low_component, high_component] {
        let covariance = &model.covariances()[component];
        assert_abs_diff_eq!(covariance[[0, 0]], 0.116_001, epsilon = 1.0e-6);
        assert_abs_diff_eq!(covariance[[0, 1]], -0.066, epsilon = 1.0e-6);
        assert_abs_diff_eq!(covariance[[1, 0]], -0.066, epsilon = 1.0e-6);
        assert_abs_diff_eq!(covariance[[1, 1]], 0.176_001, epsilon = 1.0e-6);
    }

    let query = array![[-3.0, -2.8], [3.1, 2.9]];
    let predictions = model.predict(query.view()).unwrap();
    assert_eq!(predictions[0], low_component);
    assert_eq!(predictions[1], high_component);

    let scores = model.score_samples(query.view()).unwrap();
    assert_abs_diff_eq!(scores[0], -0.501_435_78, epsilon = 1.0e-6);
    assert_abs_diff_eq!(scores[1], -0.746_762_57, epsilon = 1.0e-6);
}

#[test]
fn predict_probabilities_are_normalized() {
    let model = GaussianMixture::new(2)
        .unwrap()
        .fit(separated_dataset().view())
        .unwrap();

    let probabilities = model
        .predict_probabilities(separated_dataset().view())
        .unwrap();
    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-9);
        assert!(row.iter().all(|&value| (0.0..=1.0).contains(&value)));
    }
}

#[test]
fn is_deterministic_for_a_fixed_seed() {
    let estimator = GaussianMixture::new(2).unwrap().with_seed(11);
    let first = estimator.fit(separated_dataset().view()).unwrap();
    let second = estimator.fit(separated_dataset().view()).unwrap();

    assert_eq!(first, second);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let estimator = GaussianMixture::new(3)
        .unwrap()
        .with_max_iterations(50)
        .unwrap()
        .with_tolerance(1.0e-6)
        .unwrap()
        .with_reg_covar(1.0e-5)
        .unwrap()
        .with_seed(7);
    assert_eq!(estimator.n_components(), 3);
    assert_eq!(estimator.max_iterations(), 50);
    assert_abs_diff_eq!(estimator.tolerance(), 1.0e-6);
    assert_abs_diff_eq!(estimator.reg_covar(), 1.0e-5);
    assert_eq!(estimator.seed(), 7);

    let default = GaussianMixture::new(2).unwrap();
    assert_eq!(default.max_iterations(), 100);
    assert_abs_diff_eq!(default.tolerance(), 1.0e-3);
    assert_abs_diff_eq!(default.reg_covar(), 1.0e-6);

    assert_eq!(
        GaussianMixture::new(0).unwrap_err(),
        MlError::InvalidComponentCount(0)
    );
    assert_eq!(
        GaussianMixture::new(2)
            .unwrap()
            .with_max_iterations(0)
            .unwrap_err(),
        MlError::InvalidMaxIterations(0)
    );
    assert_eq!(
        GaussianMixture::new(2)
            .unwrap()
            .with_tolerance(0.0)
            .unwrap_err(),
        MlError::InvalidTolerance(0.0)
    );
    assert_eq!(
        GaussianMixture::new(2)
            .unwrap()
            .with_reg_covar(-1.0)
            .unwrap_err(),
        MlError::InvalidRegularization(-1.0)
    );
}

#[test]
fn rejects_more_components_than_samples() {
    let records = array![[0.0], [1.0]];
    assert_eq!(
        GaussianMixture::new(3)
            .unwrap()
            .fit(records.view())
            .unwrap_err(),
        MlError::InsufficientSamples {
            required: 3,
            actual: 2,
        }
    );
}

#[test]
fn validates_prediction_features() {
    let model = GaussianMixture::new(2)
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
    let estimator = GaussianMixture::new(2).unwrap().with_seed(3);
    let model = estimator.fit(separated_dataset().view()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: GaussianMixture = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedGaussianMixture =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.n_components(), restored_model.n_components());
    assert_eq!(model.n_iterations(), restored_model.n_iterations());
    assert_eq!(model.converged(), restored_model.converged());
    // Compared with a tolerance rather than `assert_eq!`: `serde_json`'s text
    // round trip for `f64` is not guaranteed bit-exact in every case.
    assert_abs_diff_eq!(
        model.log_likelihood(),
        restored_model.log_likelihood(),
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(
        model.means()[[0, 0]],
        restored_model.means()[[0, 0]],
        epsilon = 1.0e-9
    );
}
