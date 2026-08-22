use std::collections::BTreeMap;

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use super::Fold;
use machlearn_core::core::{MlError, Result};

/// Configures stratified K-fold cross-validation partitions.
///
/// Every observation appears in exactly one test fold. For each class, test
/// fold counts differ by at most one. Total test fold sizes also differ by at
/// most one, including when several classes have remainders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    seed: u64,
}

impl StratifiedKFold {
    /// Creates an ordered stratified K-fold configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_splits` is less than two.
    pub const fn new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(MlError::InvalidFoldCount { n_splits });
        }
        Ok(Self {
            n_splits,
            shuffle: false,
            seed: 42,
        })
    }

    /// Enables or disables shuffling observations within each class.
    #[must_use]
    pub const fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the deterministic shuffle seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the number of folds.
    #[must_use]
    pub const fn n_splits(self) -> usize {
        self.n_splits
    }

    /// Returns whether observations are shuffled within each class.
    #[must_use]
    pub const fn shuffle(self) -> bool {
        self.shuffle
    }

    /// Returns the configured shuffle seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Builds stratified training/test index partitions for `targets`.
    ///
    /// Labels are processed in sorted order. Without shuffling, observations
    /// retain their original order within each class. With shuffling enabled,
    /// only the observations within a class are shuffled, so class balance is
    /// unaffected.
    ///
    /// # Errors
    ///
    /// Returns an error when there are fewer targets than folds.
    pub fn split<Label: Ord>(self, targets: &[Label]) -> Result<Vec<Fold>> {
        let n_samples = targets.len();
        if n_samples < self.n_splits {
            return Err(MlError::InsufficientSamples {
                required: self.n_splits,
                actual: n_samples,
            });
        }

        let mut classes: BTreeMap<&Label, Vec<usize>> = BTreeMap::new();
        for (index, label) in targets.iter().enumerate() {
            classes.entry(label).or_default().push(index);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut test_indices = vec![Vec::new(); self.n_splits];
        let mut next_fold = 0;
        for class_indices in classes.values_mut() {
            if self.shuffle {
                class_indices.shuffle(&mut rng);
            }
            for (offset, &sample_index) in class_indices.iter().enumerate() {
                test_indices[(next_fold + offset) % self.n_splits].push(sample_index);
            }
            next_fold = (next_fold + class_indices.len()) % self.n_splits;
        }

        Ok(test_indices
            .into_iter()
            .map(|indices| Fold::from_test_indices(n_samples, indices))
            .collect())
    }
}
