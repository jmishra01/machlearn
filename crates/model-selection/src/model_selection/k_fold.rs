use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use machlearn_core::core::{MlError, Result};

/// Indices for one cross-validation training/test partition.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fold {
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
}

impl Fold {
    pub(super) fn from_test_indices(n_samples: usize, test_indices: Vec<usize>) -> Self {
        let mut is_test = vec![false; n_samples];
        for &index in &test_indices {
            is_test[index] = true;
        }
        let train_indices = (0..n_samples).filter(|&index| !is_test[index]).collect();
        Self {
            train_indices,
            test_indices,
        }
    }

    /// Returns indices used for fitting in this fold.
    #[must_use]
    pub fn train_indices(&self) -> &[usize] {
        &self.train_indices
    }

    /// Returns indices used for evaluation in this fold.
    #[must_use]
    pub fn test_indices(&self) -> &[usize] {
        &self.test_indices
    }

    /// Returns the number of training observations.
    #[must_use]
    pub fn train_size(&self) -> usize {
        self.train_indices.len()
    }

    /// Returns the number of test observations.
    #[must_use]
    pub fn test_size(&self) -> usize {
        self.test_indices.len()
    }
}

/// Configures K-fold cross-validation partitions.
///
/// Fold sizes differ by at most one observation. When the sample count is not
/// divisible by the fold count, earlier test folds receive one extra sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    seed: u64,
}

impl KFold {
    /// Creates an ordered K-fold configuration.
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

    /// Enables or disables shuffling before partitioning.
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

    /// Returns whether observations are shuffled before partitioning.
    #[must_use]
    pub const fn shuffle(self) -> bool {
        self.shuffle
    }

    /// Returns the configured shuffle seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Builds all training/test index partitions.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_samples` is smaller than the number of folds.
    pub fn split(self, n_samples: usize) -> Result<Vec<Fold>> {
        if n_samples < self.n_splits {
            return Err(MlError::InsufficientSamples {
                required: self.n_splits,
                actual: n_samples,
            });
        }

        let mut indices: Vec<_> = (0..n_samples).collect();
        if self.shuffle {
            indices.shuffle(&mut ChaCha8Rng::seed_from_u64(self.seed));
        }

        let base_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;
        let mut folds = Vec::with_capacity(self.n_splits);
        let mut test_start = 0;
        for fold_index in 0..self.n_splits {
            let test_size = base_size + usize::from(fold_index < remainder);
            let test_end = test_start + test_size;
            let test_indices = indices[test_start..test_end].to_vec();
            let mut train_indices = Vec::with_capacity(n_samples - test_size);
            train_indices.extend_from_slice(&indices[..test_start]);
            train_indices.extend_from_slice(&indices[test_end..]);
            folds.push(Fold {
                train_indices,
                test_indices,
            });
            test_start = test_end;
        }
        Ok(folds)
    }
}
