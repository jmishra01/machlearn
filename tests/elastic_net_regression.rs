//! Integration tests for Elastic Net (combined L1/L2) linear regression.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, ElasticNetRegression, KFold, MlError, cross_validate, mean_squared_error,
};
use ndarray::array;

fn correlated_dataset() -> Dataset<f64> {
    Dataset::new(
        array![
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 2.0],
            [5.0, 0.0, 0.0],
            [6.0, 1.0, 4.0],
        ],
        array![3.1, 4.8, 7.05, 9.0, 10.9, 13.15],
    )
    .unwrap()
}

#[test]
fn matches_a_reference_solution() {
    // Reference values confirmed against
    // `sklearn.linear_model.ElasticNet(alpha=0.5, l1_ratio=0.3, tol=1e-10,
    // max_iter=100000)` fitted on the same data.
    let model = ElasticNetRegression::new(0.5, 0.3)
        .unwrap()
        .with_tolerance(1.0e-10)
        .unwrap()
        .with_max_iterations(100_000)
        .unwrap()
        .fit(&correlated_dataset())
        .unwrap();

    assert_abs_diff_eq!(
        model.coefficients()[0],
        1.744_451_889_878_415_4,
        epsilon = 1.0e-8
    );
    assert_abs_diff_eq!(model.coefficients()[1], 0.0, epsilon = 1.0e-9);
    assert_abs_diff_eq!(
        model.coefficients()[2],
        0.052_914_319_462_078_9,
        epsilon = 1.0e-8
    );
    assert_abs_diff_eq!(model.intercept(), 1.806_227_852_988_748_8, epsilon = 1.0e-7);

    let prediction = model.predict(array![[2.5, 0.5, 1.0]].view()).unwrap();
    assert_abs_diff_eq!(prediction[0], 6.220_271_897_146_866, epsilon = 1.0e-7);
}

#[test]
fn l1_ratio_one_matches_lasso_sparsity() {
    use machlearn::LassoRegression;

    let dataset = correlated_dataset();
    let lasso = LassoRegression::new(0.5).unwrap().fit(&dataset).unwrap();
    let elastic_net = ElasticNetRegression::new(0.5, 1.0)
        .unwrap()
        .fit(&dataset)
        .unwrap();

    for (lasso_coef, elastic_net_coef) in
        lasso.coefficients().iter().zip(elastic_net.coefficients())
    {
        assert_abs_diff_eq!(lasso_coef, elastic_net_coef, epsilon = 1.0e-8);
    }
    assert_abs_diff_eq!(lasso.intercept(), elastic_net.intercept(), epsilon = 1.0e-8);
}

#[test]
fn exposes_configuration_and_validates_parameters() {
    let default = ElasticNetRegression::default();
    assert_abs_diff_eq!(default.alpha(), 1.0);
    assert_abs_diff_eq!(default.l1_ratio(), 0.5);
    assert_eq!(default.max_iterations(), 1000);
    assert_abs_diff_eq!(default.tolerance(), 1.0e-4);
    assert!(default.fit_intercept());

    let estimator = ElasticNetRegression::new(0.5, 0.3)
        .unwrap()
        .with_intercept(false);
    assert_abs_diff_eq!(estimator.alpha(), 0.5);
    assert_abs_diff_eq!(estimator.l1_ratio(), 0.3);
    assert!(!estimator.fit_intercept());

    assert_eq!(
        ElasticNetRegression::new(-1.0, 0.5).unwrap_err(),
        MlError::InvalidRegularization(-1.0)
    );
    assert_eq!(
        ElasticNetRegression::new(1.0, 1.5).unwrap_err(),
        MlError::InvalidL1Ratio(1.5)
    );
    assert_eq!(
        ElasticNetRegression::new(1.0, -0.1).unwrap_err(),
        MlError::InvalidL1Ratio(-0.1)
    );
    assert!(matches!(
        ElasticNetRegression::new(1.0, f64::NAN),
        Err(MlError::InvalidL1Ratio(value)) if value.is_nan()
    ));
}

#[test]
fn validates_prediction_features_and_targets() {
    let model = ElasticNetRegression::new(1.0, 0.5)
        .unwrap()
        .fit(&correlated_dataset())
        .unwrap();

    assert_eq!(
        model.predict(array![[1.0, 2.0]].view()).unwrap_err(),
        MlError::MismatchedFeatureCount {
            expected: 3,
            actual: 2,
        }
    );
    assert_eq!(
        model
            .predict(array![[f64::NAN, 0.0, 0.0]].view())
            .unwrap_err(),
        MlError::NonFiniteFeature { row: 0, column: 0 }
    );

    let non_finite_targets = Dataset::new(array![[0.0], [1.0]], array![0.0, f64::NAN]).unwrap();
    assert_eq!(
        ElasticNetRegression::new(1.0, 0.5)
            .unwrap()
            .fit(&non_finite_targets)
            .unwrap_err(),
        MlError::NonFiniteActualTarget { index: 1 }
    );
}

#[test]
fn composes_with_cross_validation() {
    let dataset = Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
    )
    .unwrap();
    let folds = KFold::new(3).unwrap().split(dataset.n_samples()).unwrap();

    let scores = cross_validate(
        &ElasticNetRegression::new(0.01, 0.5).unwrap(),
        &dataset,
        &folds,
        mean_squared_error,
    )
    .unwrap();

    assert_eq!(scores.scores().len(), 3);
    assert!(scores.scores().iter().all(|&score| score >= 0.0));
}

#[cfg(feature = "serde")]
#[test]
fn estimator_and_fitted_model_round_trip_through_serde() {
    let estimator = ElasticNetRegression::new(0.5, 0.3)
        .unwrap()
        .with_intercept(false);
    let dataset = correlated_dataset();
    let model = estimator.fit(&dataset).unwrap();

    let estimator_json = serde_json::to_string(&estimator).unwrap();
    let model_json = serde_json::to_string(&model).unwrap();
    let restored_estimator: ElasticNetRegression = serde_json::from_str(&estimator_json).unwrap();
    let restored_model: machlearn::FittedElasticNetRegression =
        serde_json::from_str(&model_json).unwrap();

    assert_eq!(estimator, restored_estimator);
    let predictions = model.predict(dataset.records()).unwrap();
    let restored_predictions = restored_model.predict(dataset.records()).unwrap();
    for (left, right) in predictions.iter().zip(restored_predictions.iter()) {
        assert_abs_diff_eq!(left, right, epsilon = 1.0e-9);
    }
}
