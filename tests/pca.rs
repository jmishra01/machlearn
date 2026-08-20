//! Integration tests for principal component analysis.

use approx::assert_abs_diff_eq;
use machlearn::{MlError, PrincipalComponentAnalysis};
use ndarray::array;

fn tutorial_dataset() -> ndarray::Array2<f64> {
    array![
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ]
}

#[test]
fn matches_a_reference_solution() {
    // Reference values confirmed against `sklearn.decomposition.PCA` fitted
    // on the same data (a textbook PCA example).
    let dataset = tutorial_dataset();
    let model = PrincipalComponentAnalysis::new()
        .fit(dataset.view())
        .unwrap();

    assert_abs_diff_eq!(model.mean()[0], 1.81, epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.mean()[1], 1.91, epsilon = 1.0e-12);

    assert_abs_diff_eq!(
        model.components()[[0, 0]],
        0.677_873_398_528_011_7,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.components()[[0, 1]],
        0.735_178_655_544_408_1,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.components()[[1, 0]],
        0.735_178_655_544_408_1,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.components()[[1, 1]],
        -0.677_873_398_528_011_7,
        epsilon = 1.0e-12
    );

    assert_abs_diff_eq!(
        model.explained_variance()[0],
        1.284_027_712_172_783_9,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.explained_variance()[1],
        0.049_083_398_938_327_25,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.explained_variance_ratio()[0],
        0.963_181_314_348_646,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.explained_variance_ratio()[1],
        0.036_818_685_651_353_995,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.explained_variance_ratio().sum(),
        1.0,
        epsilon = 1.0e-12
    );

    let query = array![[2.5, 2.4], [0.5, 0.7]];
    let transformed = model.transform(query.view()).unwrap();
    assert_abs_diff_eq!(
        transformed[[0, 0]],
        0.827_970_186_201_088_1,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(
        transformed[[0, 1]],
        0.175_115_307_046_915_72,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(
        transformed[[1, 0]],
        -1.777_580_325_280_429,
        epsilon = 1.0e-9
    );
    assert_abs_diff_eq!(
        transformed[[1, 1]],
        -0.142_857_226_544_280_68,
        epsilon = 1.0e-9
    );
}

#[test]
fn components_are_orthonormal() {
    let model = PrincipalComponentAnalysis::new()
        .fit(tutorial_dataset().view())
        .unwrap();

    for row in model.components().rows() {
        let norm_squared: f64 = row.iter().map(|value| value * value).sum();
        assert_abs_diff_eq!(norm_squared, 1.0, epsilon = 1.0e-9);
    }
    let dot_product: f64 = model
        .components()
        .row(0)
        .iter()
        .zip(model.components().row(1))
        .map(|(a, b)| a * b)
        .sum();
    assert_abs_diff_eq!(dot_product, 0.0, epsilon = 1.0e-9);
}

#[test]
fn inverse_transform_reconstructs_exactly_with_every_component() {
    let dataset = tutorial_dataset();
    let model = PrincipalComponentAnalysis::new()
        .fit(dataset.view())
        .unwrap();

    let transformed = model.transform(dataset.view()).unwrap();
    let reconstructed = model.inverse_transform(transformed.view()).unwrap();

    for (original, roundtrip) in dataset.iter().zip(reconstructed.iter()) {
        assert_abs_diff_eq!(original, roundtrip, epsilon = 1.0e-9);
    }
}

#[test]
fn limiting_components_keeps_the_leading_variance() {
    let dataset = tutorial_dataset();
    let full = PrincipalComponentAnalysis::new()
        .fit(dataset.view())
        .unwrap();
    let reduced = PrincipalComponentAnalysis::new()
        .with_n_components(Some(1))
        .unwrap()
        .fit(dataset.view())
        .unwrap();

    assert_eq!(reduced.n_components(), 1);
    assert_abs_diff_eq!(
        reduced.explained_variance()[0],
        full.explained_variance()[0],
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        reduced.explained_variance_ratio()[0],
        full.explained_variance_ratio()[0],
        epsilon = 1.0e-12
    );
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = PrincipalComponentAnalysis::default();
    assert_eq!(default.n_components(), None);

    let estimator = PrincipalComponentAnalysis::new()
        .with_n_components(Some(2))
        .unwrap();
    assert_eq!(estimator.n_components(), Some(2));

    assert_eq!(
        PrincipalComponentAnalysis::new()
            .with_n_components(Some(0))
            .unwrap_err(),
        MlError::InvalidComponentCount(0)
    );

    assert_eq!(
        PrincipalComponentAnalysis::new()
            .with_n_components(Some(5))
            .unwrap()
            .fit(tutorial_dataset().view())
            .unwrap_err(),
        MlError::TooManyComponents {
            requested: 5,
            maximum: 2,
        }
    );

    let single_sample = array![[1.0, 2.0]];
    assert_eq!(
        PrincipalComponentAnalysis::new()
            .fit(single_sample.view())
            .unwrap_err(),
        MlError::InsufficientSamples {
            required: 2,
            actual: 1,
        }
    );
}

#[test]
fn validates_transform_features() {
    let model = PrincipalComponentAnalysis::new()
        .fit(tutorial_dataset().view())
        .unwrap();

    assert_eq!(
        model.transform(array![[1.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        model.transform(array![[f64::NAN, 0.0]].view()).unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = PrincipalComponentAnalysis::new()
        .with_n_components(Some(1))
        .unwrap();
    let model = estimator.fit(tutorial_dataset().view()).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: PrincipalComponentAnalysis =
        serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedPrincipalComponentAnalysis =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    assert_eq!(model.n_components(), restored_model.n_components());
    // Compared with a tolerance rather than `assert_eq!`: `serde_json`'s text
    // round trip for `f64` is not guaranteed bit-exact in every case.
    assert_abs_diff_eq!(model.mean()[0], restored_model.mean()[0], epsilon = 1.0e-12);
    assert_abs_diff_eq!(model.mean()[1], restored_model.mean()[1], epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        model.components()[[0, 0]],
        restored_model.components()[[0, 0]],
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        model.explained_variance()[0],
        restored_model.explained_variance()[0],
        epsilon = 1.0e-12
    );
}
