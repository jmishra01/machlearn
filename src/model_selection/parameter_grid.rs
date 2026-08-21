use std::collections::BTreeMap;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::core::{MlError, Result};

/// A dynamically typed hyperparameter candidate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterValue {
    /// Boolean candidate.
    Boolean(bool),
    /// Signed integer candidate.
    Integer(i64),
    /// Unsigned integer candidate.
    Unsigned(u64),
    /// Finite floating-point candidate.
    Float(f64),
    /// Owned text candidate.
    Text(String),
}

impl ParameterValue {
    /// Returns the Boolean value, or `None` for another variant.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the signed integer value, or `None` for another variant.
    #[must_use]
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the unsigned integer value, or `None` for another variant.
    #[must_use]
    pub const fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the floating-point value, or `None` for another variant.
    #[must_use]
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the text value, or `None` for another variant.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            _ => None,
        }
    }
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i32> for ParameterValue {
    fn from(value: i32) -> Self {
        Self::Integer(i64::from(value))
    }
}

impl From<i64> for ParameterValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<u32> for ParameterValue {
    fn from(value: u32) -> Self {
        Self::Unsigned(u64::from(value))
    }
}

impl From<u64> for ParameterValue {
    fn from(value: u64) -> Self {
        Self::Unsigned(value)
    }
}

impl From<usize> for ParameterValue {
    fn from(value: usize) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let value = value as u64;
        Self::Unsigned(value)
    }
}

impl From<f32> for ParameterValue {
    fn from(value: f32) -> Self {
        Self::Float(f64::from(value))
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

/// One concrete hyperparameter assignment.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterSet {
    values: BTreeMap<String, ParameterValue>,
}

impl ParameterSet {
    /// Returns a parameter value by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ParameterValue> {
        self.values.get(name)
    }

    /// Iterates over assignments in lexicographic name order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &ParameterValue)> {
        self.values
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Returns the number of assignments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this is the default empty assignment.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// A deterministic Cartesian product of named hyperparameter candidates.
///
/// Parameter names are expanded in lexicographic order. Candidate values retain
/// their supplied order. An empty grid expands to one empty [`ParameterSet`],
/// representing an estimator's default configuration.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterGrid {
    parameters: BTreeMap<String, Vec<ParameterValue>>,
}

impl ParameterGrid {
    /// Creates an empty grid representing one default configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            parameters: BTreeMap::new(),
        }
    }

    /// Adds a named candidate sequence using builder syntax.
    ///
    /// # Errors
    ///
    /// Returns an error for a blank or duplicate name, an empty candidate
    /// sequence, or a non-finite floating-point candidate.
    pub fn with_parameter<Name, Values, Value>(mut self, name: Name, values: Values) -> Result<Self>
    where
        Name: Into<String>,
        Values: IntoIterator<Item = Value>,
        Value: Into<ParameterValue>,
    {
        self.insert_parameter(name, values)?;
        Ok(self)
    }

    /// Adds a named candidate sequence to this grid.
    ///
    /// # Errors
    ///
    /// Returns an error for a blank or duplicate name, an empty candidate
    /// sequence, or a non-finite floating-point candidate.
    pub fn insert_parameter<Name, Values, Value>(
        &mut self,
        name: Name,
        values: Values,
    ) -> Result<()>
    where
        Name: Into<String>,
        Values: IntoIterator<Item = Value>,
        Value: Into<ParameterValue>,
    {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(MlError::InvalidParameterName);
        }
        if self.parameters.contains_key(&name) {
            return Err(MlError::DuplicateParameter { name });
        }

        let values: Vec<_> = values.into_iter().map(Into::into).collect();
        if values.is_empty() {
            return Err(MlError::EmptyParameterValues { name });
        }
        if let Some(value_index) = values
            .iter()
            .position(|value| matches!(value, ParameterValue::Float(number) if !number.is_finite()))
        {
            return Err(MlError::NonFiniteParameterValue { name, value_index });
        }
        self.parameters.insert(name, values);
        Ok(())
    }

    /// Returns the number of named parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Returns whether the grid contains no named parameters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Returns the number of concrete assignments in the Cartesian product.
    ///
    /// # Errors
    ///
    /// Returns an error when the product exceeds `usize`.
    pub fn combination_count(&self) -> Result<usize> {
        self.parameters.values().try_fold(1_usize, |count, values| {
            count
                .checked_mul(values.len())
                .ok_or(MlError::ParameterGridTooLarge)
        })
    }

    /// Expands the deterministic Cartesian product into concrete assignments.
    ///
    /// # Errors
    ///
    /// Returns an error when the product size overflows or its storage cannot
    /// be reserved.
    pub fn combinations(&self) -> Result<Vec<ParameterSet>> {
        self.combination_count()?;
        let mut combinations = vec![ParameterSet::default()];
        for (name, values) in &self.parameters {
            let capacity = combinations
                .len()
                .checked_mul(values.len())
                .ok_or(MlError::ParameterGridTooLarge)?;
            let mut expanded = Vec::new();
            expanded
                .try_reserve_exact(capacity)
                .map_err(|_| MlError::ParameterGridTooLarge)?;
            for combination in &combinations {
                for value in values {
                    let mut assignment = combination.clone();
                    assignment.values.insert(name.clone(), value.clone());
                    expanded.push(assignment);
                }
            }
            combinations = expanded;
        }
        Ok(combinations)
    }

    /// Draws `n_iter` random concrete assignments, choosing each
    /// parameter's value independently and uniformly from its candidate
    /// list on every draw.
    ///
    /// Draws are with replacement: the same assignment may repeat, and
    /// distinct draws may coincide even without repeats when the grid is
    /// small. An empty grid draws `n_iter` copies of the empty
    /// [`ParameterSet`], matching [`Self::combinations`]'s convention that
    /// it represents one default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_iter` is zero.
    pub fn sample_combinations(&self, n_iter: usize, seed: u64) -> Result<Vec<ParameterSet>> {
        if n_iter == 0 {
            return Err(MlError::InvalidSearchIterations(n_iter));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut combinations = Vec::with_capacity(n_iter);
        for _draw in 0..n_iter {
            let mut assignment = ParameterSet::default();
            for (name, values) in &self.parameters {
                let index = rng.random_range(0..values.len());
                assignment
                    .values
                    .insert(name.clone(), values[index].clone());
            }
            combinations.push(assignment);
        }
        Ok(combinations)
    }
}
