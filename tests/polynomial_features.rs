//! Integration tests for polynomial and interaction feature expansion.

use machlearn::{MlError, PolynomialFeatures};
use ndarray::array;

#[test]
fn matches_sklearn_default_expansion() {
    // Reference column order and values confirmed against
    // `sklearn.preprocessing.PolynomialFeatures(degree=2,
    // include_bias=True)` fitted on the same data.
    let records = array![[2.0, 3.0], [1.0, 5.0]];
    let fitted = PolynomialFeatures::new(2)
        .unwrap()
        .fit(records.view())
        .unwrap();

    assert_eq!(
        fitted.combinations(),
        &[vec![], vec![0], vec![1], vec![0, 0], vec![0, 1], vec![1, 1]]
    );
    assert_eq!(fitted.n_output_features(), 6);

    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(
        transformed,
        array![
            [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [1.0, 1.0, 5.0, 1.0, 5.0, 25.0],
        ]
    );
}

#[test]
fn matches_sklearn_interaction_only_expansion() {
    // Reference confirmed against
    // `sklearn.preprocessing.PolynomialFeatures(degree=2,
    // include_bias=False, interaction_only=True)` fitted on the same data.
    let records = array![[2.0, 3.0], [1.0, 5.0]];
    let fitted = PolynomialFeatures::new(2)
        .unwrap()
        .with_include_bias(false)
        .with_interaction_only(true)
        .fit(records.view())
        .unwrap();

    assert_eq!(fitted.combinations(), &[vec![0], vec![1], vec![0, 1]]);
    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(transformed, array![[2.0, 3.0, 6.0], [1.0, 5.0, 5.0]]);
}

#[test]
fn matches_sklearn_degree_three_expansion() {
    // Reference confirmed against
    // `sklearn.preprocessing.PolynomialFeatures(degree=3,
    // include_bias=False)` fitted on the same data.
    let records = array![[2.0, 3.0]];
    let fitted = PolynomialFeatures::new(3)
        .unwrap()
        .with_include_bias(false)
        .fit(records.view())
        .unwrap();

    assert_eq!(
        fitted.combinations(),
        &[
            vec![0],
            vec![1],
            vec![0, 0],
            vec![0, 1],
            vec![1, 1],
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![1, 1, 1],
        ]
    );
    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(
        transformed,
        array![[2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0]]
    );
}

#[test]
fn a_single_feature_produces_its_powers() {
    let records = array![[2.0], [3.0]];
    let fitted = PolynomialFeatures::new(4)
        .unwrap()
        .with_include_bias(false)
        .fit(records.view())
        .unwrap();

    assert_eq!(fitted.n_output_features(), 4);
    let transformed = fitted.transform(records.view()).unwrap();
    assert_eq!(
        transformed,
        array![[2.0, 4.0, 8.0, 16.0], [3.0, 9.0, 27.0, 81.0]]
    );
}

#[test]
fn exposes_configuration_and_validates_degree() {
    let default = PolynomialFeatures::default();
    assert_eq!(default.degree(), 2);
    assert!(default.include_bias());
    assert!(!default.interaction_only());

    let configured = PolynomialFeatures::new(3)
        .unwrap()
        .with_include_bias(false)
        .with_interaction_only(true);
    assert_eq!(configured.degree(), 3);
    assert!(!configured.include_bias());
    assert!(configured.interaction_only());

    assert_eq!(
        PolynomialFeatures::new(0).unwrap_err(),
        MlError::InvalidDegree(0)
    );
}

#[test]
fn validates_transform_features() {
    let records = array![[1.0, 2.0]];
    let fitted = PolynomialFeatures::new(2)
        .unwrap()
        .fit(records.view())
        .unwrap();

    assert_eq!(
        fitted.transform(array![[1.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        fitted
            .transform(array![[f64::NAN, 0.0]].view())
            .unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn fitted_expander_round_trips_through_serde() {
    let records = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = PolynomialFeatures::new(2)
        .unwrap()
        .fit(records.view())
        .unwrap();

    let json = serde_json::to_string(&fitted).unwrap();
    let restored: machlearn::FittedPolynomialFeatures = serde_json::from_str(&json).unwrap();

    assert_eq!(fitted, restored);
}
