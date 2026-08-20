//! Benchmarks for representative fit/predict paths across estimator
//! families. Run with `cargo bench`; save a comparison baseline with
//! `cargo bench -- --save-baseline <name>`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use machlearn::{
    AdaBoostClassifier, BernoulliNaiveBayes, DBSCAN, Dataset, DecisionTreeClassifier,
    DecisionTreeRegressor, ElasticNetRegression, GaussianMixture, GradientBoostingClassifier,
    GradientBoostingRegressor, KMeans, KNeighborsClassifier, LassoRegression,
    LinearDiscriminantAnalysis, LinearRegression, LogisticRegression, MultinomialNaiveBayes,
    RandomForestClassifier, RidgeRegression, ScoreDirection, mean_squared_error,
    permutation_importance,
};
use ndarray::{Array1, Array2};

fn synthetic_regression_dataset(n_samples: usize, n_features: usize) -> Dataset<f64> {
    let records = Array2::from_shape_fn((n_samples, n_features), |(row, column)| {
        #[allow(clippy::cast_precision_loss)]
        let value = ((row * 7 + column * 3) % 97) as f64 * 0.1;
        value
    });
    #[allow(clippy::cast_precision_loss)]
    let targets = Array1::from_shape_fn(n_samples, |row| (row % 50) as f64);
    Dataset::new(records, targets).unwrap()
}

fn synthetic_classification_dataset(n_samples: usize, n_features: usize) -> Dataset<u8> {
    let records = Array2::from_shape_fn((n_samples, n_features), |(row, column)| {
        #[allow(clippy::cast_precision_loss)]
        let value = ((row * 7 + column * 3) % 97) as f64 * 0.1;
        value
    });
    let targets = Array1::from_shape_fn(n_samples, |row| u8::from(row % 2 == 0));
    Dataset::new(records, targets).unwrap()
}

fn bench_linear_regression_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(500, 10);
    c.bench_function("linear_regression_fit_500x10", |b| {
        b.iter(|| LinearRegression::new().fit(black_box(&dataset)).unwrap());
    });
}

fn bench_ridge_regression_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(500, 10);
    let estimator = RidgeRegression::new(1.0).unwrap();
    c.bench_function("ridge_regression_fit_500x10", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_lasso_regression_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(500, 10);
    let estimator = LassoRegression::new(0.1).unwrap();
    c.bench_function("lasso_regression_fit_500x10", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_elastic_net_regression_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(500, 10);
    let estimator = ElasticNetRegression::new(0.1, 0.5).unwrap();
    c.bench_function("elastic_net_regression_fit_500x10", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_logistic_regression_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(500, 10);
    c.bench_function("logistic_regression_fit_500x10", |b| {
        b.iter(|| LogisticRegression::new().fit(black_box(&dataset)).unwrap());
    });
}

fn bench_multinomial_naive_bayes_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(500, 10);
    c.bench_function("multinomial_naive_bayes_fit_500x10", |b| {
        b.iter(|| {
            MultinomialNaiveBayes::new()
                .fit(black_box(&dataset))
                .unwrap()
        });
    });
}

fn bench_bernoulli_naive_bayes_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(500, 10);
    c.bench_function("bernoulli_naive_bayes_fit_500x10", |b| {
        b.iter(|| BernoulliNaiveBayes::new().fit(black_box(&dataset)).unwrap());
    });
}

fn bench_lda_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(500, 10);
    c.bench_function("linear_discriminant_analysis_fit_500x10", |b| {
        b.iter(|| {
            LinearDiscriminantAnalysis::new()
                .fit(black_box(&dataset))
                .unwrap()
        });
    });
}

fn bench_knn_predict(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(500, 10);
    let model = KNeighborsClassifier::new(5).unwrap().fit(&dataset).unwrap();
    let query = synthetic_classification_dataset(100, 10).into_parts().0;
    c.bench_function("knn_classifier_predict_500train_100query", |b| {
        b.iter(|| model.predict(black_box(query.view())).unwrap());
    });
}

fn bench_decision_tree_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(500, 10);
    c.bench_function("decision_tree_classifier_fit_500x10", |b| {
        b.iter(|| {
            DecisionTreeClassifier::new()
                .fit(black_box(&dataset))
                .unwrap()
        });
    });
}

fn bench_random_forest_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(300, 8);
    let estimator = RandomForestClassifier::new().with_n_estimators(20).unwrap();
    c.bench_function("random_forest_classifier_fit_300x8_20trees", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_gradient_boosting_regressor_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(300, 8);
    let estimator = GradientBoostingRegressor::new()
        .with_n_estimators(20)
        .unwrap();
    c.bench_function("gradient_boosting_regressor_fit_300x8_20trees", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_gradient_boosting_classifier_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(300, 8);
    let estimator = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .unwrap();
    c.bench_function("gradient_boosting_classifier_fit_300x8_20trees", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_adaboost_classifier_fit(c: &mut Criterion) {
    let dataset = synthetic_classification_dataset(300, 8);
    let estimator = AdaBoostClassifier::new().with_n_estimators(20).unwrap();
    c.bench_function("adaboost_classifier_fit_300x8_20rounds", |b| {
        b.iter(|| estimator.fit(black_box(&dataset)).unwrap());
    });
}

fn bench_permutation_importance(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(300, 8);
    let model = DecisionTreeRegressor::new().fit(&dataset).unwrap();
    c.bench_function("permutation_importance_300x8_10repeats", |b| {
        b.iter(|| {
            permutation_importance(
                &model,
                black_box(&dataset),
                mean_squared_error,
                ScoreDirection::Minimize,
                10,
                0,
            )
            .unwrap()
        });
    });
}

fn bench_kmeans_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(500, 5);
    let records = dataset.records();
    let estimator = KMeans::new(4).unwrap();
    c.bench_function("kmeans_fit_500x5_4clusters", |b| {
        b.iter(|| estimator.fit(black_box(records)).unwrap());
    });
}

fn bench_dbscan_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(200, 5);
    let records = dataset.records();
    let estimator = DBSCAN::new(5.0, 5).unwrap();
    c.bench_function("dbscan_fit_200x5", |b| {
        b.iter(|| estimator.fit(black_box(records)).unwrap());
    });
}

fn bench_gaussian_mixture_fit(c: &mut Criterion) {
    let dataset = synthetic_regression_dataset(200, 5);
    let records = dataset.records();
    let estimator = GaussianMixture::new(3).unwrap();
    c.bench_function("gaussian_mixture_fit_200x5_3components", |b| {
        b.iter(|| estimator.fit(black_box(records)).unwrap());
    });
}

criterion_group!(
    benches,
    bench_linear_regression_fit,
    bench_ridge_regression_fit,
    bench_lasso_regression_fit,
    bench_elastic_net_regression_fit,
    bench_logistic_regression_fit,
    bench_multinomial_naive_bayes_fit,
    bench_bernoulli_naive_bayes_fit,
    bench_lda_fit,
    bench_knn_predict,
    bench_decision_tree_fit,
    bench_random_forest_fit,
    bench_gradient_boosting_regressor_fit,
    bench_gradient_boosting_classifier_fit,
    bench_adaboost_classifier_fit,
    bench_permutation_importance,
    bench_kmeans_fit,
    bench_dbscan_fit,
    bench_gaussian_mixture_fit,
);
criterion_main!(benches);
