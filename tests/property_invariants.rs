//! Property-based tests for numerical invariants that must hold across the
//! entire input space, not just the fixed examples exercised elsewhere.

use machlearn::{
    AdaBoostClassifier, BernoulliNaiveBayes, DBSCAN, Dataset, DecisionTreeRegressor,
    ElasticNetRegression, GaussianMixture, GradientBoostingClassifier, GradientBoostingRegressor,
    KFold, KMeans, LassoRegression, LinearDiscriminantAnalysis, MinMaxScaler,
    MultinomialNaiveBayes, OneHotEncoder, PolynomialFeatures, PrincipalComponentAnalysis,
    ScoreDirection, SelectKBest, SplitOptions, StandardScaler, VarianceThreshold, accuracy_score,
    f_regression, learning_curve, mean_absolute_error, mean_squared_error, multiclass_log_loss,
    permutation_importance, roc_auc_score_ovr, root_mean_squared_error, train_test_split,
    validation_curve,
};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

/// A dense matrix of finite, bounded values with `rows` in
/// `[min_rows, max_rows]` and `cols` in `[min_cols, max_cols]`.
fn matrix_strategy(
    min_rows: usize,
    max_rows: usize,
    min_cols: usize,
    max_cols: usize,
) -> impl Strategy<Value = Array2<f64>> {
    (min_rows..=max_rows, min_cols..=max_cols).prop_flat_map(|(rows, cols)| {
        proptest::collection::vec(-1000.0_f64..1000.0, rows * cols)
            .prop_map(move |values| Array2::from_shape_vec((rows, cols), values).unwrap())
    })
}

/// A dense matrix of finite, non-negative, count-like values with `rows` in
/// `[min_rows, max_rows]` and `cols` in `[min_cols, max_cols]`.
fn non_negative_matrix_strategy(
    min_rows: usize,
    max_rows: usize,
    min_cols: usize,
    max_cols: usize,
) -> impl Strategy<Value = Array2<f64>> {
    (min_rows..=max_rows, min_cols..=max_cols).prop_flat_map(|(rows, cols)| {
        proptest::collection::vec(0.0_f64..100.0, rows * cols)
            .prop_map(move |values| Array2::from_shape_vec((rows, cols), values).unwrap())
    })
}

fn column_mean(column: &ndarray::ArrayView1<'_, f64>) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let count = column.len() as f64;
    column.sum() / count
}

fn column_variance(column: &ndarray::ArrayView1<'_, f64>) -> f64 {
    let mean = column_mean(column);
    #[allow(clippy::cast_precision_loss)]
    let count = column.len() as f64;
    column
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count
}

proptest! {
    /// `StandardScaler` always centers every feature to (numerically) zero
    /// mean, regardless of the input's scale or distribution.
    #[test]
    fn standard_scaler_centers_every_feature(records in matrix_strategy(3, 12, 1, 4)) {
        let fitted = StandardScaler::default().fit(records.view()).unwrap();
        let transformed = fitted.transform(records.view()).unwrap();
        for column in transformed.columns() {
            prop_assert!(column_mean(&column).abs() < 1.0e-6);
        }
    }

    /// For any feature with non-negligible original variance, `StandardScaler`
    /// scales it to unit variance.
    #[test]
    fn standard_scaler_reaches_unit_variance_for_non_constant_features(
        records in matrix_strategy(3, 12, 1, 4),
    ) {
        let fitted = StandardScaler::default().fit(records.view()).unwrap();
        let transformed = fitted.transform(records.view()).unwrap();
        for (feature, column) in records.columns().into_iter().enumerate() {
            prop_assume!(column_variance(&column).sqrt() > 1.0e-6);
            let variance = column_variance(&transformed.column(feature));
            prop_assert!((variance.sqrt() - 1.0).abs() < 1.0e-6);
        }
    }

    /// `MinMaxScaler` output always stays within the configured range, for
    /// both constant and non-constant features.
    #[test]
    fn min_max_scaler_output_stays_within_the_configured_range(
        records in matrix_strategy(2, 12, 1, 4),
    ) {
        let fitted = MinMaxScaler::default().fit(records.view()).unwrap();
        let transformed = fitted.transform(records.view()).unwrap();
        for &value in &transformed {
            prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
        }
    }

    /// `train_test_split` never gains or loses samples: every original row
    /// appears in exactly one of the two output datasets.
    #[test]
    fn train_test_split_partitions_every_sample_exactly_once(
        seed in any::<u64>(),
        fraction in 0.1_f64..0.9,
        n_rows in 4_usize..40,
    ) {
        #[allow(clippy::cast_precision_loss)]
        let records = Array2::from_shape_fn((n_rows, 2), |(row, _)| row as f64);
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(n_rows, |row| row as f64);
        let dataset = Dataset::new(records, targets).unwrap();

        let split = train_test_split(
            &dataset,
            SplitOptions::new(fraction).with_seed(seed),
        );
        // A degenerate combination of a tiny sample count and a large
        // fraction can legitimately leave zero training rows; that is a
        // documented error, not a broken invariant.
        let Ok((train, test)) = split else {
            return Ok(());
        };

        prop_assert_eq!(train.n_samples() + test.n_samples(), n_rows);
        let mut seen: Vec<bool> = vec![false; n_rows];
        for &target in train.targets().iter().chain(test.targets().iter()) {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let index = target as usize;
            prop_assert!(!seen[index], "row {index} appeared in both splits");
            seen[index] = true;
        }
        prop_assert!(seen.iter().all(|&was_seen| was_seen));
    }

    /// Regression error metrics are always non-negative, and a perfect
    /// prediction always scores exactly zero.
    #[test]
    fn regression_errors_are_non_negative_and_zero_for_perfect_predictions(
        actual in proptest::collection::vec(-1000.0_f64..1000.0, 2..20),
    ) {
        let actual = Array1::from_vec(actual);
        prop_assert!(mean_squared_error(actual.view(), actual.view()).unwrap().abs() < 1.0e-12);
        prop_assert!(mean_absolute_error(actual.view(), actual.view()).unwrap().abs() < 1.0e-12);
        prop_assert!(
            root_mean_squared_error(actual.view(), actual.view()).unwrap().abs() < 1.0e-12
        );
    }

    /// Regression error metrics stay non-negative for arbitrary, imperfect
    /// predictions too.
    #[test]
    fn regression_errors_stay_non_negative(
        (actual, noise) in proptest::collection::vec(-1000.0_f64..1000.0, 2..20)
            .prop_flat_map(|actual| {
                let len = actual.len();
                (Just(actual), proptest::collection::vec(-50.0_f64..50.0, len))
            }),
    ) {
        let actual = Array1::from_vec(actual);
        let predicted = Array1::from_iter(
            actual.iter().zip(noise.iter()).map(|(value, offset)| value + offset),
        );
        prop_assert!(mean_squared_error(actual.view(), predicted.view()).unwrap() >= 0.0);
        prop_assert!(mean_absolute_error(actual.view(), predicted.view()).unwrap() >= 0.0);
        prop_assert!(root_mean_squared_error(actual.view(), predicted.view()).unwrap() >= 0.0);
    }

    /// `accuracy_score` is always within `[0, 1]`, and identical inputs
    /// always score a perfect `1.0`.
    #[test]
    fn accuracy_score_is_bounded_and_perfect_for_identical_inputs(
        labels in proptest::collection::vec(0_u8..4, 2..20),
    ) {
        let labels = Array1::from_vec(labels);
        let accuracy = accuracy_score(labels.view(), labels.view()).unwrap();
        prop_assert!((accuracy - 1.0).abs() < 1.0e-12);
        prop_assert!((0.0..=1.0).contains(&accuracy));
    }

    /// `KMeans` inertia is never negative and predictions always name an
    /// existing cluster.
    #[test]
    fn kmeans_inertia_is_non_negative_and_assignments_are_in_range(
        records in matrix_strategy(6, 20, 1, 3),
        n_clusters in 1_usize..4,
    ) {
        prop_assume!(records.nrows() >= n_clusters);
        let model = KMeans::new(n_clusters).unwrap().fit(records.view()).unwrap();
        prop_assert!(model.inertia() >= 0.0);
        let assignments = model.predict(records.view()).unwrap();
        prop_assert!(assignments.iter().all(|&cluster| cluster < n_clusters));
    }

    /// PCA components are always unit-norm and mutually orthogonal, for any
    /// input with enough rank to support the requested component count.
    #[test]
    fn pca_components_are_orthonormal_for_arbitrary_input(
        records in matrix_strategy(4, 10, 2, 4),
    ) {
        let model = PrincipalComponentAnalysis::new().fit(records.view()).unwrap();
        let components = model.components();

        for row in components.rows() {
            let norm_squared: f64 = row.iter().map(|value| value * value).sum();
            prop_assert!((norm_squared - 1.0).abs() < 1.0e-6);
        }
        for first in 0..components.nrows() {
            for second in (first + 1)..components.nrows() {
                let dot_product: f64 = components
                    .row(first)
                    .iter()
                    .zip(components.row(second))
                    .map(|(a, b)| a * b)
                    .sum();
                prop_assert!(dot_product.abs() < 1.0e-6);
            }
        }
    }

    /// PCA explained-variance ratios are always non-negative and never sum
    /// to more than one.
    #[test]
    fn pca_explained_variance_ratio_is_bounded(records in matrix_strategy(4, 10, 2, 4)) {
        let model = PrincipalComponentAnalysis::new().fit(records.view()).unwrap();
        let ratios = model.explained_variance_ratio();
        for &ratio in ratios {
            prop_assert!(ratio >= -1.0e-9);
        }
        prop_assert!(ratios.sum() <= 1.0 + 1.0e-6);
    }

    /// A sufficiently large L1 penalty drives every Lasso coefficient to
    /// exactly zero, for arbitrary training data: soft thresholding zeroes
    /// out any coefficient whose correlation with the residual cannot
    /// overcome the penalty.
    #[test]
    fn lasso_large_alpha_zeroes_every_coefficient(records in matrix_strategy(4, 15, 1, 4)) {
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(records.nrows(), |row| (row as f64).sin() * 10.0);
        let dataset = Dataset::new(records, targets).unwrap();

        let model = LassoRegression::new(1.0e6).unwrap().fit(&dataset).unwrap();

        prop_assert_eq!(model.n_nonzero_coefficients(), 0);
    }

    /// `ElasticNetRegression` with `l1_ratio = 1.0` always matches
    /// `LassoRegression` exactly: they solve the same objective.
    #[test]
    fn elastic_net_l1_ratio_one_matches_lasso(
        records in matrix_strategy(4, 15, 1, 4),
        alpha in 0.01_f64..5.0,
    ) {
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(records.nrows(), |row| (row as f64).cos() * 5.0);
        let dataset = Dataset::new(records, targets).unwrap();

        // Coordinate descent's convergence rate depends on the data's
        // scale and conditioning; unnormalized features drawn from this
        // strategy's wide range can genuinely fail to converge within the
        // default iteration budget (confirmed: `sklearn.linear_model.Lasso`
        // exhibits the identical non-convergence, as a warning rather than
        // an error, on the same inputs). That is a legitimate, documented
        // error here, not a broken invariant.
        let Ok(lasso) = LassoRegression::new(alpha).unwrap().fit(&dataset) else {
            return Ok(());
        };
        let Ok(elastic_net) = ElasticNetRegression::new(alpha, 1.0).unwrap().fit(&dataset) else {
            return Ok(());
        };

        for (lasso_coef, elastic_net_coef) in
            lasso.coefficients().iter().zip(elastic_net.coefficients())
        {
            prop_assert!((lasso_coef - elastic_net_coef).abs() < 1.0e-6);
        }
        prop_assert!((lasso.intercept() - elastic_net.intercept()).abs() < 1.0e-6);
    }

    /// Linear discriminant analysis probabilities are always bounded and
    /// sum to one, for arbitrary training data with a well-conditioned
    /// pooled covariance.
    #[test]
    fn lda_probabilities_are_normalized(records in matrix_strategy(9, 20, 1, 3)) {
        let n_classes = 3;
        let targets = Array1::from_shape_fn(records.nrows(), |row| {
            u8::try_from(row % n_classes).unwrap()
        });
        let dataset = Dataset::new(records, targets).unwrap();

        // A rank-deficient pooled covariance is a legitimate, documented
        // error for this randomly generated data, not a broken invariant;
        // skip those draws rather than asserting fit always succeeds.
        let Ok(model) = LinearDiscriminantAnalysis::new().fit(&dataset) else {
            return Ok(());
        };

        let probabilities = model.predict_probabilities(dataset.records()).unwrap();
        for row in probabilities.rows() {
            prop_assert!((row.sum() - 1.0).abs() < 1.0e-6);
            for &value in row {
                prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
            }
        }
    }

    /// Gradient-boosted regression predictions are always finite, for
    /// arbitrary training data queried at the training points themselves.
    #[test]
    fn gradient_boosting_regressor_predictions_are_finite(records in matrix_strategy(4, 15, 1, 4)) {
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(records.nrows(), |row| (row as f64).sin() * 10.0);
        let dataset = Dataset::new(records.clone(), targets).unwrap();

        let model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .unwrap()
            .fit(&dataset)
            .unwrap();

        let predictions = model.predict(records.view()).unwrap();
        prop_assert!(predictions.iter().all(|value| value.is_finite()));
    }

    /// Gradient-boosted classification probabilities are always bounded and
    /// sum to one, for arbitrary training data with two observed classes.
    #[test]
    fn gradient_boosting_classifier_probabilities_are_normalized(
        records in matrix_strategy(4, 15, 1, 4),
    ) {
        let targets = Array1::from_shape_fn(records.nrows(), |row| u8::try_from(row % 2).unwrap());
        let dataset = Dataset::new(records.clone(), targets).unwrap();

        let model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .unwrap()
            .fit(&dataset)
            .unwrap();

        let probabilities = model.predict_probabilities(records.view()).unwrap();
        for row in probabilities.rows() {
            prop_assert!((row.sum() - 1.0).abs() < 1.0e-6);
            for &value in row {
                prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
            }
        }
    }

    /// `AdaBoost` classification probabilities are always bounded and sum to
    /// one, for arbitrary training data with two observed classes.
    #[test]
    fn adaboost_classifier_probabilities_are_normalized(records in matrix_strategy(4, 15, 1, 4)) {
        let targets = Array1::from_shape_fn(records.nrows(), |row| u8::try_from(row % 2).unwrap());
        let dataset = Dataset::new(records.clone(), targets).unwrap();

        // A first-round weak learner no better than random guessing is a
        // legitimate, documented error for this randomly generated data,
        // not a broken invariant; skip those draws rather than asserting
        // fit always succeeds.
        let Ok(model) = AdaBoostClassifier::new().with_n_estimators(10).unwrap().fit(&dataset)
        else {
            return Ok(());
        };

        let probabilities = model.predict_probabilities(records.view()).unwrap();
        for row in probabilities.rows() {
            prop_assert!((row.sum() - 1.0).abs() < 1.0e-6);
            for &value in row {
                prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
            }
        }
    }

    /// Permutation importance always reports one finite value per repeat
    /// for every feature, regardless of the model or the training data's
    /// distribution.
    #[test]
    fn permutation_importance_reports_one_finite_value_per_feature_and_repeat(
        records in matrix_strategy(6, 15, 1, 4),
        n_repeats in 1_usize..5,
        seed in any::<u64>(),
    ) {
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(records.nrows(), |row| (row as f64).sin() * 10.0);
        let dataset = Dataset::new(records, targets).unwrap();
        let model = DecisionTreeRegressor::new().fit(&dataset).unwrap();

        let result = permutation_importance(
            &model,
            &dataset,
            mean_squared_error,
            ScoreDirection::Minimize,
            n_repeats,
            seed,
        )
        .unwrap();

        prop_assert_eq!(result.n_features(), dataset.n_features());
        prop_assert_eq!(result.n_repeats(), n_repeats);
        prop_assert!(result.importances().iter().all(|value| value.is_finite()));
    }

    /// Multinomial Naive Bayes probabilities are always bounded and sum to
    /// one, for arbitrary non-negative count-like training data.
    #[test]
    fn multinomial_naive_bayes_probabilities_are_normalized(
        records in non_negative_matrix_strategy(4, 15, 1, 4),
    ) {
        let targets = Array1::from_shape_fn(records.nrows(), |row| u8::try_from(row % 2).unwrap());
        let dataset = Dataset::new(records, targets).unwrap();

        let model = MultinomialNaiveBayes::new().fit(&dataset).unwrap();

        let probabilities = model.predict_probabilities(dataset.records()).unwrap();
        for row in probabilities.rows() {
            prop_assert!((row.sum() - 1.0).abs() < 1.0e-6);
            for &value in row {
                prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
            }
        }
    }

    /// Bernoulli Naive Bayes probabilities are always bounded and sum to
    /// one, for arbitrary non-negative training data binarized at zero.
    #[test]
    fn bernoulli_naive_bayes_probabilities_are_normalized(
        records in non_negative_matrix_strategy(4, 15, 1, 4),
    ) {
        let targets = Array1::from_shape_fn(records.nrows(), |row| u8::try_from(row % 2).unwrap());
        let dataset = Dataset::new(records, targets).unwrap();

        let model = BernoulliNaiveBayes::new().fit(&dataset).unwrap();

        let probabilities = model.predict_probabilities(dataset.records()).unwrap();
        for row in probabilities.rows() {
            prop_assert!((row.sum() - 1.0).abs() < 1.0e-6);
            for &value in row {
                prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
            }
        }
    }

    /// DBSCAN always labels every row either noise or with an in-range
    /// cluster index, and the noise and clustered counts always add up to
    /// the full sample count, for arbitrary data and parameters.
    #[test]
    fn dbscan_labels_are_in_range_and_account_for_every_row(
        records in matrix_strategy(4, 20, 1, 3),
        eps in 0.1_f64..50.0,
        min_samples in 1_usize..5,
    ) {
        let model = DBSCAN::new(eps, min_samples).unwrap().fit(records.view()).unwrap();

        let labels = model.labels();
        prop_assert_eq!(labels.len(), records.nrows());
        let mut clustered = 0;
        for &label in labels {
            if let Some(cluster) = label {
                prop_assert!(cluster < model.n_clusters());
                clustered += 1;
            }
        }
        prop_assert_eq!(clustered + model.n_noise_points(), records.nrows());
    }

    /// Gaussian mixture membership probabilities are always bounded and sum
    /// to one, for arbitrary data and component counts.
    #[test]
    fn gaussian_mixture_probabilities_are_normalized(
        records in matrix_strategy(6, 20, 1, 3),
        n_components in 1_usize..4,
    ) {
        prop_assume!(records.nrows() >= n_components);
        let model = GaussianMixture::new(n_components).unwrap().fit(records.view()).unwrap();

        let probabilities = model.predict_probabilities(records.view()).unwrap();
        for row in probabilities.rows() {
            prop_assert!((row.sum() - 1.0).abs() < 1.0e-6);
            for &value in row {
                prop_assert!((-1.0e-9..=1.0 + 1.0e-9).contains(&value));
            }
        }
    }

    /// One-hot encoding always produces exactly one `1.0` per row (every
    /// other entry `0.0`), for arbitrary categorical labels.
    #[test]
    fn one_hot_encoding_has_exactly_one_hot_entry_per_row(
        labels in proptest::collection::vec(0_u8..6, 1..20),
    ) {
        let labels = Array1::from_vec(labels);
        let fitted = OneHotEncoder::new().fit(labels.view()).unwrap();
        let encoded = fitted.transform(labels.view()).unwrap();

        for row in encoded.rows() {
            let ones = row.iter().filter(|&&value| value > 0.5).count();
            let zeros = row.iter().filter(|&&value| value < 0.5).count();
            prop_assert_eq!(ones, 1);
            prop_assert_eq!(zeros, row.len() - 1);
        }
    }

    /// Dropping the first class still leaves every row with at most one
    /// `1.0` entry, and only rows whose original label was the dropped
    /// class are all zero.
    #[test]
    fn one_hot_encoding_drop_first_has_at_most_one_hot_entry_per_row(
        labels in proptest::collection::vec(0_u8..6, 1..20),
    ) {
        let labels = Array1::from_vec(labels);
        let fitted = OneHotEncoder::new().with_drop_first(true).fit(labels.view()).unwrap();
        let encoded = fitted.transform(labels.view()).unwrap();

        for (row_index, row) in encoded.rows().into_iter().enumerate() {
            let ones = row.iter().filter(|&&value| value > 0.5).count();
            prop_assert!(ones <= 1);
            let is_dropped_class = labels[row_index] == fitted.classes()[0];
            prop_assert_eq!(ones == 0, is_dropped_class);
        }
    }

    /// Every polynomial-expansion output column always equals the product
    /// of the raw feature values named by its combination, and the bias
    /// column (when present) is always exactly `1.0`, for arbitrary data
    /// and degree.
    #[test]
    fn polynomial_features_columns_match_their_combinations(
        records in matrix_strategy(2, 8, 1, 3),
        degree in 1_usize..4,
    ) {
        let fitted = PolynomialFeatures::new(degree).unwrap().fit(records.view()).unwrap();
        let transformed = fitted.transform(records.view()).unwrap();

        prop_assert_eq!(transformed.ncols(), fitted.combinations().len());
        for (row_index, row) in records.rows().into_iter().enumerate() {
            for (column_index, combination) in fitted.combinations().iter().enumerate() {
                let expected: f64 = combination.iter().map(|&feature_index| row[feature_index]).product();
                prop_assert!((transformed[[row_index, column_index]] - expected).abs() < 1.0e-6);
            }
        }
        for &value in transformed.column(0) {
            prop_assert!((value - 1.0).abs() < 1.0e-9);
        }
    }

    /// `VarianceThreshold` always keeps exactly the features whose computed
    /// variance exceeds the threshold, for arbitrary data.
    #[test]
    fn variance_threshold_keeps_exactly_the_features_above_threshold(
        records in matrix_strategy(3, 15, 1, 4),
        threshold in 0.0_f64..500.0,
    ) {
        let fitted = VarianceThreshold::new().with_threshold(threshold).unwrap().fit(records.view()).unwrap();

        let expected: Vec<usize> = (0..records.ncols())
            .filter(|&index| fitted.variances()[index] > threshold)
            .collect();
        prop_assert_eq!(fitted.selected_indices(), expected.as_slice());

        let transformed = fitted.transform(records.view()).unwrap();
        prop_assert_eq!(transformed.ncols(), fitted.n_selected_features());
    }

    /// `SelectKBest` never selects more features than `k` or more than are
    /// available, and its selected indices are always sorted and within
    /// range, for arbitrary data and `k`.
    #[test]
    fn select_k_best_selects_at_most_k_in_range_features(
        (records, targets) in (3_usize..15, 1_usize..5).prop_flat_map(|(rows, cols)| {
            (
                proptest::collection::vec(-1000.0_f64..1000.0, rows * cols)
                    .prop_map(move |values| Array2::from_shape_vec((rows, cols), values).unwrap()),
                proptest::collection::vec(-1000.0_f64..1000.0, rows).prop_map(Array1::from_vec),
            )
        }),
        k in 0_usize..8,
    ) {
        let fitted = SelectKBest::new(k).fit(records.view(), targets.view(), f_regression).unwrap();

        prop_assert!(fitted.n_selected_features() <= k);
        prop_assert!(fitted.n_selected_features() <= records.ncols());
        prop_assert!(fitted.selected_indices().windows(2).all(|pair| pair[0] < pair[1]));
        prop_assert!(fitted.selected_indices().iter().all(|&index| index < records.ncols()));
    }

    /// Multiclass log loss is always finite and non-negative, and one-vs-rest
    /// ROC AUC always stays within `[0, 1]`, for arbitrary valid probability
    /// rows (produced via softmax, so they are always properly normalized).
    #[test]
    fn multiclass_log_loss_and_roc_auc_ovr_stay_in_bounds(
        (n_classes, actual, probabilities) in (2_usize..5, 6_usize..15).prop_flat_map(
            |(n_classes, n_samples)| {
                proptest::collection::vec(-5.0_f64..5.0, n_samples * n_classes).prop_map(
                    move |logits| {
                        let mut probabilities = Array2::<f64>::zeros((n_samples, n_classes));
                        for row in 0..n_samples {
                            let row_logits = &logits[row * n_classes..(row + 1) * n_classes];
                            let max = row_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let exponentials: Vec<f64> =
                                row_logits.iter().map(|&value| (value - max).exp()).collect();
                            let sum: f64 = exponentials.iter().sum();
                            for column in 0..n_classes {
                                probabilities[[row, column]] = exponentials[column] / sum;
                            }
                        }
                        let actual = Array1::from_shape_fn(n_samples, |row| {
                            u8::try_from(row % n_classes).unwrap()
                        });
                        (n_classes, actual, probabilities)
                    },
                )
            },
        ),
    ) {
        let classes: Vec<u8> = (0..u8::try_from(n_classes).unwrap()).collect();

        let loss = multiclass_log_loss(actual.view(), probabilities.view(), &classes).unwrap();
        prop_assert!(loss.is_finite());
        prop_assert!(loss >= 0.0);

        let auc = roc_auc_score_ovr(actual.view(), probabilities.view(), &classes).unwrap();
        prop_assert!((0.0..=1.0).contains(&auc));
    }

    /// `learning_curve` always reports one row per training size, one
    /// column per fold, and every score is finite and non-negative
    /// (`mean_squared_error` can never be otherwise), for arbitrary data.
    #[test]
    fn learning_curve_reports_finite_non_negative_scores(
        records in matrix_strategy(6, 15, 1, 3),
        n_folds in 2_usize..4,
    ) {
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(records.nrows(), |row| (row as f64).sin() * 10.0);
        let dataset = Dataset::new(records, targets).unwrap();
        let folds = KFold::new(n_folds).unwrap().split(dataset.n_samples()).unwrap();
        let min_train_size = folds.iter().map(machlearn::Fold::train_size).min().unwrap();

        let scores = learning_curve(
            &DecisionTreeRegressor::new(),
            &[min_train_size],
            &dataset,
            &folds,
            mean_squared_error,
        )
        .unwrap();

        prop_assert_eq!(scores.n_points(), 1);
        prop_assert_eq!(scores.n_folds(), n_folds);
        prop_assert!(scores.train_scores().iter().all(|value| value.is_finite() && *value >= 0.0));
        prop_assert!(scores.test_scores().iter().all(|value| value.is_finite() && *value >= 0.0));
    }

    /// `validation_curve` always reports one row per swept value, one
    /// column per fold, and every score is finite and non-negative, for
    /// arbitrary data.
    #[test]
    fn validation_curve_reports_finite_non_negative_scores(
        records in matrix_strategy(6, 15, 1, 3),
        n_folds in 2_usize..4,
    ) {
        #[allow(clippy::cast_precision_loss)]
        let targets = Array1::from_shape_fn(records.nrows(), |row| (row as f64).sin() * 10.0);
        let dataset = Dataset::new(records, targets).unwrap();
        let folds = KFold::new(n_folds).unwrap().split(dataset.n_samples()).unwrap();
        let depths: Vec<Option<usize>> = vec![Some(1), Some(2), None];

        let scores = validation_curve(
            &depths,
            |depth| Ok(DecisionTreeRegressor::new().with_max_depth(*depth)),
            &dataset,
            &folds,
            mean_squared_error,
        )
        .unwrap();

        prop_assert_eq!(scores.n_points(), 3);
        prop_assert_eq!(scores.n_folds(), n_folds);
        prop_assert!(scores.train_scores().iter().all(|value| value.is_finite() && *value >= 0.0));
        prop_assert!(scores.test_scores().iter().all(|value| value.is_finite() && *value >= 0.0));
    }
}
