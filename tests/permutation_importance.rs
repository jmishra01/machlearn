//! Integration tests for model-agnostic permutation feature importance.

use approx::assert_abs_diff_eq;
use machlearn::{
    Dataset, DecisionTreeClassifier, LinearRegression, MlError, Predict, Result, ScoreDirection,
    accuracy_score, mean_squared_error, permutation_importance,
};
use ndarray::{Array1, ArrayView2, array};

/// A model that ignores every feature except `used_column`, echoing that
/// column's value as its prediction. Because no other feature can ever
/// influence its output, permuting them must leave the score exactly
/// unchanged, regardless of which random permutation was drawn.
struct EchoColumn {
    used_column: usize,
}

impl Predict<ArrayView2<'_, f64>> for EchoColumn {
    type Output = Array1<f64>;

    fn predict(&self, features: ArrayView2<'_, f64>) -> Result<Self::Output> {
        Ok(features.column(self.used_column).to_owned())
    }
}

fn perfectly_predictable_dataset() -> Dataset<f64> {
    Dataset::new(
        array![
            [0.0, 9.0],
            [1.0, 2.0],
            [2.0, 7.0],
            [3.0, 1.0],
            [4.0, 8.0],
            [5.0, 3.0],
            [6.0, 6.0],
            [7.0, 4.0],
        ],
        array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    .unwrap()
}

#[test]
#[allow(clippy::float_cmp)]
fn an_unused_feature_always_reports_exactly_zero_importance() {
    let dataset = perfectly_predictable_dataset();
    let model = EchoColumn { used_column: 0 };

    let result = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        20,
        7,
    )
    .unwrap();

    // Column 1 never reaches the model's prediction, so shuffling it can
    // never move the score, for any of the twenty random permutations.
    for &value in result.importances().row(1) {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn a_used_feature_with_a_perfect_baseline_fit_reports_positive_importance() {
    let dataset = perfectly_predictable_dataset();
    let model = EchoColumn { used_column: 0 };

    let result = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        20,
        7,
    )
    .unwrap();

    // The baseline score is a perfect fit (zero error), so any permutation
    // that actually moves column 0 (all but an astronomically unlikely
    // identity draw, across twenty independent repeats on eight rows) must
    // increase the error, i.e. report positive importance under Minimize.
    for &value in result.importances().row(0) {
        assert!(value > 0.0);
    }
    assert!(result.importances_mean()[0] > result.importances_mean()[1]);
}

#[test]
fn importances_mean_and_std_match_a_manual_reduction_of_the_raw_values() {
    let dataset = perfectly_predictable_dataset();
    let model = EchoColumn { used_column: 0 };

    let result = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        6,
        3,
    )
    .unwrap();

    let mean = result.importances_mean();
    let std = result.importances_std();
    let importances = result.importances();
    for feature in 0..result.n_features() {
        let row = importances.row(feature);
        let expected_mean = row.sum() / f64::from(u32::try_from(row.len()).unwrap());
        assert_abs_diff_eq!(mean[feature], expected_mean, epsilon = 1.0e-12);

        let expected_variance = row
            .iter()
            .map(|value| (value - expected_mean).powi(2))
            .sum::<f64>()
            / f64::from(u32::try_from(row.len()).unwrap());
        assert_abs_diff_eq!(std[feature], expected_variance.sqrt(), epsilon = 1.0e-9);
    }
}

#[test]
fn is_deterministic_for_a_fixed_seed() {
    let dataset = perfectly_predictable_dataset();
    let model = EchoColumn { used_column: 0 };

    let first = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        5,
        11,
    )
    .unwrap();
    let second = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        5,
        11,
    )
    .unwrap();

    assert_eq!(first, second);
}

#[test]
fn a_real_regressor_ranks_the_informative_feature_above_the_noise_feature() {
    let dataset = perfectly_predictable_dataset();
    let model = LinearRegression::new().fit(&dataset).unwrap();

    let result = permutation_importance(
        &model,
        &dataset,
        mean_squared_error,
        ScoreDirection::Minimize,
        10,
        0,
    )
    .unwrap();

    assert!(result.importances_mean()[0] > result.importances_mean()[1]);
}

#[test]
fn maximize_direction_also_reports_positive_importance_for_an_informative_feature() {
    let dataset = Dataset::new(
        array![[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        array![0_u8, 0, 0, 1, 1, 1],
    )
    .unwrap();
    let model = DecisionTreeClassifier::new().fit(&dataset).unwrap();

    let result = permutation_importance(
        &model,
        &dataset,
        accuracy_score,
        ScoreDirection::Maximize,
        20,
        0,
    )
    .unwrap();

    assert!(result.importances_mean()[0] > 0.0);
}

#[test]
fn rejects_a_zero_repeat_count() {
    let dataset = perfectly_predictable_dataset();
    let model = EchoColumn { used_column: 0 };

    assert_eq!(
        permutation_importance(
            &model,
            &dataset,
            mean_squared_error,
            ScoreDirection::Minimize,
            0,
            0,
        )
        .unwrap_err(),
        MlError::InvalidRepeatCount(0)
    );
}
