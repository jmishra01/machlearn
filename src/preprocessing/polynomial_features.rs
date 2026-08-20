// `ArrayView2` is a lightweight view descriptor; accepting it by value avoids
// requiring callers to borrow a temporary view.
#![allow(clippy::needless_pass_by_value)]

use ndarray::{Array2, ArrayView2};

use crate::core::{MlError, Result, Transform, validate_feature_count, validate_features};

const DEFAULT_DEGREE: usize = 2;

/// Configures polynomial and interaction feature expansion.
///
/// Every output column is the product of a multiset of input feature
/// columns, one column per combination of feature indices of size `0` (the
/// bias, a column of ones) up through `degree`. Combinations are generated
/// with replacement (so squared and higher-power terms like `x0^2` appear)
/// unless `interaction_only` is set, which keeps only combinations of
/// distinct features (`x0 * x1`, never `x0^2`). Columns are ordered by
/// increasing degree, matching common convention: bias, then every input
/// feature, then every degree-2 combination, and so on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self {
            degree: DEFAULT_DEGREE,
            include_bias: true,
            interaction_only: false,
        }
    }
}

impl PolynomialFeatures {
    /// Creates a polynomial feature expander up to `degree`, including a
    /// bias column.
    ///
    /// # Errors
    ///
    /// Returns an error when `degree` is zero.
    pub fn new(degree: usize) -> Result<Self> {
        validate_degree(degree)?;
        Ok(Self {
            degree,
            include_bias: true,
            interaction_only: false,
        })
    }

    /// Sets whether a bias column of all ones is included.
    #[must_use]
    pub const fn with_include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Sets whether only combinations of distinct features are kept,
    /// excluding pure powers of a single feature.
    #[must_use]
    pub const fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Returns the configured maximum degree.
    #[must_use]
    pub const fn degree(self) -> usize {
        self.degree
    }

    /// Returns whether a bias column is included.
    #[must_use]
    pub const fn include_bias(self) -> bool {
        self.include_bias
    }

    /// Returns whether only distinct-feature combinations are kept.
    #[must_use]
    pub const fn interaction_only(self) -> bool {
        self.interaction_only
    }

    /// Records the input feature count and the combinations
    /// [`FittedPolynomialFeatures::transform`] will compute.
    ///
    /// # Errors
    ///
    /// Returns an error when `degree` is zero, or when features are empty
    /// or non-finite.
    pub fn fit(&self, records: ArrayView2<'_, f64>) -> Result<FittedPolynomialFeatures> {
        validate_degree(self.degree)?;
        validate_features(records)?;

        let n_features = records.ncols();
        let combinations = feature_combinations(
            n_features,
            self.degree,
            self.include_bias,
            self.interaction_only,
        );
        Ok(FittedPolynomialFeatures {
            n_features,
            combinations,
            degree: self.degree,
            include_bias: self.include_bias,
            interaction_only: self.interaction_only,
        })
    }
}

/// The feature combinations learned by [`PolynomialFeatures`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedPolynomialFeatures {
    n_features: usize,
    combinations: Vec<Vec<usize>>,
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
}

impl FittedPolynomialFeatures {
    /// Returns the number of input features seen during fitting.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of output columns [`Self::transform`] produces.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.combinations.len()
    }

    /// Returns the configured maximum degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Returns whether a bias column is included.
    #[must_use]
    pub const fn include_bias(&self) -> bool {
        self.include_bias
    }

    /// Returns whether only distinct-feature combinations are kept.
    #[must_use]
    pub const fn interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Returns the input feature indices multiplied together to produce
    /// every output column, in output-column order.
    ///
    /// The bias column (when present) is the empty combination.
    #[must_use]
    pub fn combinations(&self) -> &[Vec<usize>] {
        &self.combinations
    }

    /// Expands every row into its polynomial and interaction features.
    ///
    /// # Errors
    ///
    /// Returns an error when features are empty, non-finite, have the wrong
    /// column count, or produce a non-finite output value.
    pub fn transform(&self, records: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        validate_features(records)?;
        validate_feature_count(records.ncols(), self.n_features)?;

        let mut output = Array2::zeros((records.nrows(), self.combinations.len()));
        for (row_index, row) in records.rows().into_iter().enumerate() {
            for (column_index, combination) in self.combinations.iter().enumerate() {
                let value = combination
                    .iter()
                    .map(|&feature_index| row[feature_index])
                    .product::<f64>();
                if !value.is_finite() {
                    return Err(MlError::NonFinitePrediction { index: row_index });
                }
                output[[row_index, column_index]] = value;
            }
        }
        Ok(output)
    }
}

impl<'a> Transform<ArrayView2<'a, f64>> for FittedPolynomialFeatures {
    type Output = Array2<f64>;

    fn transform(&self, input: ArrayView2<'a, f64>) -> Result<Self::Output> {
        Self::transform(self, input)
    }
}

/// Every output column's feature-index combination, ordered by increasing
/// degree and then by [`combinations_with_replacement`]'s enumeration
/// order within a degree: `1, x0, x1, x0^2, x0 x1, x1^2, ...` for two
/// features up to degree two.
fn feature_combinations(
    n_features: usize,
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
) -> Vec<Vec<usize>> {
    let start_degree = usize::from(!include_bias);
    let mut combinations = Vec::new();
    for level in start_degree..=degree {
        if level == 0 {
            combinations.push(Vec::new());
            continue;
        }
        if interaction_only {
            combinations_without_replacement(n_features, level, &mut combinations);
        } else {
            combinations_with_replacement(n_features, level, &mut combinations);
        }
    }
    combinations
}

/// Every non-decreasing sequence of length `size` drawn from `0..n`,
/// matching Python's `itertools.combinations_with_replacement`.
fn combinations_with_replacement(n: usize, size: usize, out: &mut Vec<Vec<usize>>) {
    fn recurse(
        start: usize,
        n: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        for index in start..n {
            current.push(index);
            recurse(index, n, remaining - 1, current, out);
            current.pop();
        }
    }
    recurse(0, n, size, &mut Vec::new(), out);
}

/// Every strictly increasing sequence of length `size` drawn from `0..n`,
/// matching Python's `itertools.combinations`.
fn combinations_without_replacement(n: usize, size: usize, out: &mut Vec<Vec<usize>>) {
    fn recurse(
        start: usize,
        n: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        for index in start..n {
            current.push(index);
            recurse(index + 1, n, remaining - 1, current, out);
            current.pop();
        }
    }
    recurse(0, n, size, &mut Vec::new(), out);
}

fn validate_degree(degree: usize) -> Result<()> {
    if degree == 0 {
        return Err(MlError::InvalidDegree(degree));
    }
    Ok(())
}
