# MachLearn development tracker

Last updated: 2026-08-17

Status markers:

- `[x]` complete
- `[-]` in progress
- `[ ]` pending

Roadmap scope: this tracker uses the scikit-learn 1.9 user guide and API
reference as a capability inventory, not as a promise of one-to-one API
compatibility. MachLearn remains CPU-first and Rust-native. Items are ordered by
dependency and expected value for dense tabular workloads; sparse, text, image,
and manifold-learning support come later because they require additional data
structures and solver work.

Reference inventory:

- [scikit-learn 1.9 user guide](https://scikit-learn.org/stable/user_guide.html)
- [scikit-learn 1.9 API reference](https://scikit-learn.org/stable/api/index.html)

## 1. Project foundation

- [x] Create the Rust 2024 library crate
- [x] Define `Fit`, `Predict`, and `Transform` API contracts
- [x] Add a validated dense `Dataset<Target>` type
- [x] Add structured public errors
- [x] Implement deterministic train/test splitting
- [x] Add optional `serde` and parallel feature flags
- [x] Enable strict formatting and Clippy checks
- [x] Add runnable examples for every currently exposed capability
- [ ] Choose and add the project license files
- [x] Initialize a Git repository and continuous integration

## 2. Preprocessing

- [x] Implement `StandardScaler`
- [x] Implement `MinMaxScaler`
- [x] Add fitted transformer composition through `Pipeline`
- [x] Implement categorical label encoding
- [x] Define and implement a missing-value policy
- [x] Add train-only fitting examples that demonstrate leakage prevention

## 3. Metrics

- [x] Mean squared error, root mean squared error, and mean absolute error
- [x] R-squared score
- [x] Accuracy and confusion matrix
- [x] Precision, recall, and F1 score
- [x] Binary log loss and ROC AUC
- [x] Define behavior for empty inputs, zero divisions, and invalid probabilities

## 4. Model selection

- [x] K-fold cross-validation
- [x] Stratified K-fold cross-validation
- [x] Cross-validation scoring
- [x] Parameter grid representation
- [x] Grid search
- [x] Optional parallel fold and parameter evaluation

## 5. Linear models

- [x] Add `faer` as the internal numerical solver backend
- [x] Ordinary least-squares regression using QR or SVD
- [x] Ridge regression
- [x] Binary logistic regression
- [x] Multiclass logistic regression
- [ ] Convergence reports and configurable stopping criteria

## 6. Priority 0: estimator and data API foundations

These capabilities should land before significantly expanding the model catalog.

- [ ] Add `PredictProba` and `DecisionFunction` traits for classifiers
- [ ] Add `Score` and a typed scorer registry with greater-is-better metadata
- [ ] Add estimator parameter introspection, cloning, and nested parameter paths
- [ ] Add common fitted-model metadata: `n_features_in`, classes, feature names,
      convergence state, and training iteration count
- [ ] Define deterministic random-state handling shared by all stochastic APIs
- [ ] Add optional sample weights to estimators, metrics, and splitters
- [ ] Add multi-output target containers and classifier/regressor capabilities
- [ ] Add sparse matrix support (CSR first, then CSC) behind an optional feature
- [ ] Define warm-start and `partial_fit`/online-learning contracts
- [ ] Add reusable pairwise distance, similarity, and kernel APIs
- [ ] Add L1/L2 norms, stable statistics, probability distributions, and
      eigendecomposition/SVD solver primitives required by later algorithms
- [ ] Extend `Pipeline` to include a final predictor and expose nested parameters
- [ ] Add `ColumnTransformer`, `FeatureUnion`, and transformed-target regression
- [ ] Add feature-name propagation and column-selection metadata
- [ ] Standardize feature importance, coefficient, and component accessors

## 7. Priority 0: high-value supervised models

### Linear and generalized linear models

- [ ] Add configurable convergence reports and stopping criteria to iterative
      linear estimators
- [ ] Lasso regression with coordinate descent
- [ ] Elastic Net regression
- [ ] Ridge classifier and cross-validated Ridge/Lasso/Elastic Net variants
- [ ] Stochastic-gradient classifier and regressor with `partial_fit`
- [ ] Perceptron and passive-aggressive classifier/regressor
- [ ] Robust regression: Huber, RANSAC, and Theil-Sen
- [ ] Quantile regression
- [ ] Generalized linear regressors: Poisson, Gamma, and Tweedie
- [ ] Bayesian ridge and automatic relevance determination regression

### Nearest neighbours and probabilistic classifiers

- [ ] Brute-force nearest-neighbour search with Euclidean, Manhattan, Minkowski,
      and cosine distance
- [ ] K-nearest-neighbour classifier and regressor, including distance weights
- [ ] Radius-neighbour classifier and regressor
- [ ] KD-tree and ball-tree indices after brute-force reference behavior is stable
- [ ] Nearest-centroid classifier
- [ ] Gaussian Naive Bayes
- [ ] Multinomial, Bernoulli, Complement, and Categorical Naive Bayes
- [ ] Linear and quadratic discriminant analysis

### Trees and ensembles

- [ ] CART decision-tree classifier with Gini and entropy/log-loss criteria
- [ ] CART decision-tree regressor with squared, absolute, Poisson, and Friedman
      error criteria
- [ ] Tree controls: depth, leaf sizes, feature subsampling, class weights, and
      cost-complexity pruning
- [ ] Tree feature importance and exportable tree structure
- [ ] Random-forest classifier and regressor
- [ ] Extra-trees classifier and regressor
- [ ] Bagging classifier and regressor
- [ ] AdaBoost classifier and regressor
- [ ] Gradient-boosting classifier and regressor
- [ ] Histogram gradient boosting, including missing-value handling
- [ ] Voting and stacking classifier/regressor meta-estimators

### Kernel and neural models

- [ ] Linear support-vector classifier and regressor
- [ ] Kernel SVC, SVR, NuSVC, NuSVR, and one-class SVM
- [ ] Kernel ridge regression
- [ ] Multi-layer perceptron classifier and regressor
- [ ] Gaussian-process regressor, classifier, and composable kernels

## 8. Priority 0: core unsupervised learning

### Clustering

- [ ] K-means with k-means++ and random initialization
- [ ] Mini-batch and bisecting K-means
- [ ] DBSCAN
- [ ] Agglomerative clustering and feature agglomeration
- [ ] Mean Shift
- [ ] OPTICS and HDBSCAN
- [ ] Spectral clustering and affinity propagation
- [ ] BIRCH
- [ ] Clustering outputs: labels, centers, inertia, core samples, and hierarchy

### Decomposition and mixtures

- [ ] Principal component analysis with explained variance and inverse transform
- [ ] Incremental PCA and truncated SVD
- [ ] Non-negative matrix factorization
- [ ] FastICA and factor analysis
- [ ] Sparse PCA and dictionary learning
- [ ] Kernel PCA
- [ ] Latent Dirichlet allocation after sparse count matrices exist
- [ ] Gaussian mixture models with AIC/BIC and sampling
- [ ] Bayesian Gaussian mixtures

### Density, anomaly, and covariance estimation

- [ ] Kernel density estimation
- [ ] Local Outlier Factor
- [ ] Isolation Forest
- [ ] Elliptic Envelope and robust covariance
- [ ] Empirical, shrunk, Ledoit-Wolf, and OAS covariance estimators
- [ ] Graphical Lasso covariance estimation

## 9. Priority 1: preprocessing, imputation, and feature engineering

- [ ] Robust, max-absolute, and per-sample normalization scalers
- [ ] Binarizer and polynomial/interaction features
- [ ] One-hot, ordinal, label-binarizing, multilabel-binarizing, and target encoders
- [ ] Unknown-category and infrequent-category policies for categorical encoders
- [ ] Quantile and power transforms
- [ ] Discretization and spline basis transformers
- [ ] Missing-value indicator and K-nearest-neighbour imputation
- [ ] Iterative multivariate imputation
- [ ] Variance-threshold feature selection
- [ ] Univariate feature selection: ANOVA F, chi-squared, correlation, and mutual
      information scores
- [ ] Model-based, recursive, and sequential feature selection
- [ ] Gaussian and sparse random projections
- [ ] Kernel approximations: Nystroem, random Fourier features, and additive
      chi-squared maps
- [ ] Pairwise-distance and nearest-neighbour graph transformers
- [ ] Text feature extraction: count, hashing, and TF-IDF vectorizers after sparse
      matrices and a tokenizer contract exist
- [ ] Dictionary and feature hashing for structured records
- [ ] Image patch extraction and image-to-graph helpers (long term)

## 10. Priority 1: model selection and evaluation

### Data splitting and resampling

- [ ] Shuffle split and stratified shuffle split
- [ ] Repeated K-fold and repeated stratified K-fold
- [ ] Group K-fold, stratified group K-fold, group shuffle split, and
      leave-one-group-out
- [ ] Time-series split with gap and expanding-window controls
- [ ] Leave-one-out and leave-p-out
- [ ] Predefined splits and user-supplied fold indices

### Search, validation, and prediction

- [ ] Randomized hyperparameter search with typed distributions
- [ ] Successive-halving grid and randomized search
- [ ] Multiple metrics, refit policies, error handling, and complete CV result tables
- [ ] Cross-validated predictions and probability predictions
- [ ] Learning curves, validation curves, and permutation-test scores
- [ ] Nested cross-validation helpers
- [ ] Tuned classification decision thresholds
- [ ] Baseline/dummy classifier and regressor

### Metrics and scoring

- [ ] Multiclass and multilabel log loss and ROC AUC
- [ ] Precision-recall curve, ROC curve, DET curve, average precision, and threshold
      metric utilities
- [ ] Balanced accuracy, top-k accuracy, Brier score, F-beta, Jaccard, Hamming,
      hinge, Cohen kappa, and Matthews correlation
- [ ] Ranking metrics: DCG, NDCG, coverage error, and label-ranking loss/AP
- [ ] Regression: explained variance, max error, median absolute error, MAPE,
      MSLE/RMSLE, pinball loss, and D-squared scores
- [ ] Poisson, Gamma, and Tweedie deviance metrics
- [ ] Clustering: Rand/adjusted Rand, mutual information, homogeneity,
      completeness, V-measure, Fowlkes-Mallows, silhouette, Calinski-Harabasz,
      and Davies-Bouldin scores
- [ ] Pairwise distances, similarities, kernels, and distance-metric registry
- [ ] Multi-output aggregation and sample-weight behavior for every metric

## 11. Priority 2: advanced estimators and meta-estimators

- [ ] Multiclass strategies: one-vs-rest, one-vs-one, error-correcting output codes,
      and output-code metadata
- [ ] Multi-output classifier/regressor and classifier chains
- [ ] Probability calibration with sigmoid and isotonic methods
- [ ] Isotonic regression
- [ ] Semi-supervised self-training, label propagation, and label spreading
- [ ] Partial least squares regression/canonical/SVD and canonical correlation
- [ ] Least-angle regression, Lasso-LARS, and orthogonal matching pursuit
- [ ] Multi-task Lasso and Elastic Net
- [ ] Neighbourhood components analysis
- [ ] Density-ratio or novelty-scoring interfaces shared by anomaly models

## 12. Priority 3: manifold learning and specialized unsupervised models

- [ ] Isomap and locally linear embedding variants
- [ ] Spectral embedding
- [ ] Multidimensional scaling
- [ ] t-SNE
- [ ] Spectral biclustering and co-clustering
- [ ] Restricted Boltzmann machine
- [ ] Trustworthiness and biclustering evaluation metrics

## 13. Inspection and interpretability

- [ ] Permutation feature importance for any scorer/model pair
- [ ] Partial dependence and individual conditional expectation data APIs
- [ ] Calibration-curve data
- [ ] Prediction-error, confusion-matrix, ROC, precision-recall, and DET display data
      (rendering should remain outside the core crate)
- [ ] Estimator HTML/text descriptions or a Rust-native model summary

## 14. Dataset utilities and synthetic generators

- [ ] Optional CSV dataset loading with schema and missing-value configuration
- [ ] Bundled small benchmark datasets with license metadata
- [ ] Synthetic classification, regression, blobs, moons, circles, and covariance
      generators
- [ ] Dataset shuffling, resampling, class-weight, and sample-weight helpers
- [ ] Evaluate Arrow or Polars interoperability

## 15. Persistence and integrations

- [ ] Design a versioned model serialization envelope
- [ ] Model round-trip tests with the `serde` feature
- [ ] Evaluate Python bindings with PyO3
- [ ] Evaluate WebAssembly support
- [ ] Add an optional ONNX export path for supported fitted estimators
- [ ] Document model persistence security, versioning, and reproducibility limits

## 16. Performance, correctness, and release gates

- [ ] Reference-result tests against established implementations
- [ ] Property tests for every numerical invariant
- [ ] Criterion benchmark suite and saved baselines
- [ ] Benchmark time and peak memory against scikit-learn on fixed datasets
- [ ] Add deterministic fixtures produced by a pinned scikit-learn version
- [ ] Add fuzz tests for parsers, parameter validation, and malformed model data
- [ ] Test reproducibility across thread counts and supported platforms
- [ ] Define numerical-tolerance and floating-point stability policies
- [ ] Profile and optimize allocation-heavy fit/predict paths
- [ ] Add optional SIMD where benchmarks demonstrate a material improvement
- [ ] Test default, no-default, and all-feature configurations in CI
- [ ] Audit dependencies and licenses
- [ ] Complete API documentation and runnable examples
- [ ] Define minimum-supported-Rust-version testing
- [ ] Publish the `0.1.0` release

## Recommended delivery sequence

- [ ] **Milestone A — dependable tabular baseline:** estimator API foundations,
      Lasso/Elastic Net, brute-force KNN, Gaussian Naive Bayes, CART, K-means,
      PCA, common encoders, expanded metrics, and randomized search
- [ ] **Milestone B — competitive tabular toolkit:** sparse matrices, random forests,
      extra trees, gradient boosting, SVMs, remaining Naive Bayes models,
      `ColumnTransformer`, feature selection, group/time-series CV, and inspection
- [ ] **Milestone C — scalable and online:** SGD/online APIs, mini-batch algorithms,
      partial fitting, histogram gradient boosting, sparse text features, and
      performance/parallelism work
- [ ] **Milestone D — advanced statistical models:** Gaussian processes, mixtures,
      covariance models, calibration, semi-supervised learning, and robust/GLM
      regression
- [ ] **Milestone E — specialized parity:** manifold learning, biclustering,
      dictionary learning, image graph helpers, and remaining long-tail utilities

## Current milestone exit criteria

The linear-model milestone is complete when:

- `faer` is isolated behind an internal numerical-solver module;
- ordinary least-squares and ridge regression match reference solutions;
- binary and multiclass logistic regression expose stable probability predictions;
- iterative solvers report convergence and honor stopping criteria;
- fitted models reject incompatible or non-finite prediction inputs;
- strict Clippy, default-feature tests, and all-feature tests pass.
