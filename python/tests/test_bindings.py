"""End-to-end tests for the machlearn PyO3 bindings.

These exercise the compiled extension (install it first with
``maturin develop`` from the ``python/`` directory), covering every bound
class/function so the bindings don't silently rot as the Rust API evolves.
"""

import numpy as np
import pytest

import machlearn

REG_RECORDS = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
REG_TARGETS = np.array([9.0, 8.0, 19.0, 18.0])  # = 2 * col0 + 3 * col1 + 1

CLF_RECORDS = np.array(
    [[0.1, 1.0], [0.2, 1.1], [5.0, 6.0], [5.2, 6.3], [0.15, 0.9], [5.1, 6.1]]
)
CLF_TARGETS = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)


def test_dataset_and_train_test_split():
    dataset = machlearn.Dataset(REG_RECORDS, REG_TARGETS)
    assert dataset.shape == (4, 2)
    np.testing.assert_allclose(dataset.records, REG_RECORDS)
    np.testing.assert_allclose(dataset.targets, REG_TARGETS)

    train, test = machlearn.train_test_split(dataset, 0.25, seed=7, shuffle=True)
    assert train.shape[0] + test.shape[0] == 4


def test_linear_regression():
    dataset = machlearn.Dataset(REG_RECORDS, REG_TARGETS)
    model = machlearn.LinearRegression()
    model.fit(dataset)
    predictions = model.predict(REG_RECORDS)
    assert machlearn.r2_score(REG_TARGETS, predictions) > 0.99

    with pytest.raises(RuntimeError):
        machlearn.LinearRegression().predict(REG_RECORDS)


@pytest.mark.parametrize(
    "make_model",
    [
        lambda: machlearn.RidgeRegression(alpha=0.5),
        lambda: machlearn.LassoRegression(alpha=0.1),
        lambda: machlearn.ElasticNetRegression(alpha=0.1, l1_ratio=0.5),
        lambda: machlearn.DecisionTreeRegressor(),
        lambda: machlearn.RandomForestRegressor(n_estimators=10),
        lambda: machlearn.KNeighborsRegressor(3),
        lambda: machlearn.GradientBoostingRegressor(n_estimators=20, max_depth=2),
    ],
)
def test_dataset_based_regressors_fit_predict(make_model):
    dataset = machlearn.Dataset(REG_RECORDS, REG_TARGETS)
    model = make_model()
    model.fit(dataset)
    predictions = model.predict(REG_RECORDS)
    assert predictions.shape == (4,)
    assert np.all(np.isfinite(predictions))


@pytest.mark.parametrize(
    "make_model",
    [
        lambda: machlearn.LogisticRegression(),
        lambda: machlearn.DecisionTreeClassifier(max_depth=3),
        lambda: machlearn.RandomForestClassifier(n_estimators=10, max_depth=3),
        lambda: machlearn.KNeighborsClassifier(3),
        lambda: machlearn.GaussianNaiveBayes(),
        lambda: machlearn.MultinomialNaiveBayes(),
        lambda: machlearn.LinearDiscriminantAnalysis(),
        lambda: machlearn.GradientBoostingClassifier(n_estimators=20, max_depth=2),
        lambda: machlearn.AdaBoostClassifier(n_estimators=10),
    ],
)
def test_raw_array_classifiers_fit_predict(make_model):
    model = make_model()
    model.fit(CLF_RECORDS, CLF_TARGETS)
    predictions = model.predict(CLF_RECORDS)
    assert machlearn.accuracy_score(CLF_TARGETS, predictions) == 1.0


def test_bernoulli_naive_bayes():
    # Bernoulli NB binarizes features at a threshold before fitting; the
    # default threshold of 0.0 would binarize every value in CLF_RECORDS
    # (all positive) to 1, making the classes indistinguishable. Use a
    # threshold that actually separates the two clusters.
    model = machlearn.BernoulliNaiveBayes(binarize=2.5)
    model.fit(CLF_RECORDS, CLF_TARGETS)
    predictions = model.predict(CLF_RECORDS)
    assert machlearn.accuracy_score(CLF_TARGETS, predictions) == 1.0


def test_logistic_regression_extras():
    model = machlearn.LogisticRegression()
    model.fit(CLF_RECORDS, CLF_TARGETS)
    assert model.classes == (0, 1)
    proba = model.predict_proba(CLF_RECORDS)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.isfinite(model.intercept)
    assert model.coefficients.shape == (2,)


def test_kmeans():
    model = machlearn.KMeans(n_clusters=2)
    model.fit(CLF_RECORDS)
    labels = model.predict(CLF_RECORDS)
    assert len(set(labels.tolist())) == 2


def test_dbscan():
    model = machlearn.DBSCAN(eps=1.0, min_samples=2)
    model.fit(CLF_RECORDS)
    assert model.n_clusters() == 2
    labels = model.labels()
    assert labels.shape == (6,)


def test_gaussian_mixture():
    model = machlearn.GaussianMixture(n_components=2)
    model.fit(CLF_RECORDS)
    labels = model.predict(CLF_RECORDS)
    proba = model.predict_proba(CLF_RECORDS)
    assert labels.shape == (6,)
    assert proba.shape == (6, 2)


def test_pca():
    records = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0], [2.0, 1.0, 0.0]]
    )
    pca = machlearn.PCA(n_components=2)
    projected = pca.fit_transform(records)
    assert projected.shape == (4, 2)
    assert pca.explained_variance_ratio.shape == (2,)
    assert 0.0 <= pca.explained_variance_ratio.sum() <= 1.0 + 1e-9


def test_standard_scaler():
    scaler = machlearn.StandardScaler()
    scaled = scaler.fit_transform(REG_RECORDS)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)

    with pytest.raises(RuntimeError):
        machlearn.StandardScaler().transform(REG_RECORDS)


def test_simple_imputer():
    raw = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    imputer = machlearn.SimpleImputer.mean()
    clean = imputer.fit_transform(raw)
    assert not np.isnan(clean).any()
    np.testing.assert_allclose(imputer.fill_values, [3.0, 3.0])


def test_polynomial_features():
    poly = machlearn.PolynomialFeatures(degree=2)
    expanded = poly.fit_transform(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert expanded.shape == (2, 6)


def test_label_encoder():
    encoder = machlearn.LabelEncoder()
    encoder.fit(["cat", "dog", "cat", "bird"])
    codes = encoder.transform(["dog", "bird"])
    assert codes.tolist() == [2, 0]
    assert encoder.classes == ["bird", "cat", "dog"]
    assert encoder.inverse_transform(codes.tolist()) == ["dog", "bird"]


def test_one_hot_encoder():
    encoder = machlearn.OneHotEncoder()
    encoder.fit(["cat", "dog", "bird"])
    onehot = encoder.transform(["dog", "cat"])
    assert onehot.shape == (2, 3)
    assert encoder.classes == ["bird", "cat", "dog"]


def test_classification_metrics():
    actual = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    predicted = np.array([0, 0, 1, 0, 0, 1], dtype=np.int64)
    proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9])

    assert 0.0 <= machlearn.precision_score(actual, predicted) <= 1.0
    assert 0.0 <= machlearn.recall_score(actual, predicted) <= 1.0
    assert 0.0 <= machlearn.f1_score(actual, predicted) <= 1.0
    assert machlearn.roc_auc_score(actual, proba, 1) == 1.0

    counts, classes = machlearn.confusion_matrix(actual, predicted)
    assert counts.shape == (2, 2)
    assert classes.tolist() == [0, 1]


def test_regression_metrics():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 1.9, 3.2])
    assert machlearn.mean_squared_error(actual, predicted) == pytest.approx(
        0.02, abs=1e-9
    )
    assert machlearn.r2_score(actual, predicted) > 0.9


def test_kfold():
    folds = machlearn.KFold(n_splits=3, shuffle=True, seed=7).split(9)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert len(train_idx) + len(test_idx) == 9
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())


def test_stratified_kfold():
    targets = [0, 0, 0, 1, 1, 1, 1, 1, 1]
    folds = machlearn.StratifiedKFold(n_splits=3, seed=7).split(targets)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert len(train_idx) + len(test_idx) == 9


def test_manual_cross_validation_with_kfold():
    records = np.arange(9.0).reshape(9, 1)
    targets = records.flatten() * 2 + 1

    scores = []
    for train_idx, test_idx in machlearn.KFold(n_splits=3, seed=7).split(9):
        model = machlearn.LinearRegression()
        model.fit(machlearn.Dataset(records[train_idx], targets[train_idx]))
        predictions = model.predict(records[test_idx])
        scores.append(machlearn.r2_score(targets[test_idx], predictions))

    assert all(score > 0.99 for score in scores)
