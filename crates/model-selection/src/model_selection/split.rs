use ndarray::Axis;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use machlearn_core::core::{Dataset, MlError, Result};

/// Configuration for a train/test split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitOptions {
    test_fraction: f64,
    seed: u64,
    shuffle: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            test_fraction: 0.2,
            seed: 42,
            shuffle: true,
        }
    }
}

impl SplitOptions {
    /// Creates split options with a test fraction and sensible defaults.
    ///
    /// The fraction is validated by [`train_test_split`].
    #[must_use]
    pub const fn new(test_fraction: f64) -> Self {
        Self {
            test_fraction,
            seed: 42,
            shuffle: true,
        }
    }

    /// Sets the deterministic random seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enables or disables shuffling before splitting.
    #[must_use]
    pub const fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Returns the requested test fraction.
    #[must_use]
    pub const fn test_fraction(self) -> f64 {
        self.test_fraction
    }

    /// Returns the configured random seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Returns whether rows will be shuffled before splitting.
    #[must_use]
    pub const fn shuffle(self) -> bool {
        self.shuffle
    }
}

/// Splits a dataset into non-overlapping training and test datasets.
///
/// The number of test samples is `ceil(n_samples * test_fraction)`. When
/// shuffling is disabled, training receives the first rows and testing receives
/// the final rows. Shuffled splits are deterministic for a given crate version,
/// seed, and dataset.
///
/// # Errors
///
/// Returns an error for a non-finite fraction, a fraction outside `(0, 1)`, or
/// a dataset containing fewer than two samples.
pub fn train_test_split<Target: Clone>(
    dataset: &Dataset<Target>,
    options: SplitOptions,
) -> Result<(Dataset<Target>, Dataset<Target>)> {
    if !options.test_fraction.is_finite()
        || options.test_fraction <= 0.0
        || options.test_fraction >= 1.0
    {
        return Err(MlError::InvalidTestFraction(options.test_fraction));
    }

    let sample_count = dataset.n_samples();
    if sample_count < 2 {
        return Err(MlError::InsufficientSamples {
            required: 2,
            actual: sample_count,
        });
    }

    // The value is finite, positive, and below `sample_count` after validation.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    let test_count = ((sample_count as f64) * options.test_fraction).ceil() as usize;
    let train_count = sample_count - test_count;
    if train_count == 0 {
        return Err(MlError::InsufficientSamples {
            required: test_count + 1,
            actual: sample_count,
        });
    }

    let mut indices: Vec<usize> = (0..sample_count).collect();
    if options.shuffle {
        indices.shuffle(&mut ChaCha8Rng::seed_from_u64(options.seed));
    }

    let (train_indices, test_indices) = indices.split_at(train_count);
    let train = select(dataset, train_indices)?;
    let test = select(dataset, test_indices)?;
    Ok((train, test))
}

fn select<Target: Clone>(dataset: &Dataset<Target>, indices: &[usize]) -> Result<Dataset<Target>> {
    let records = dataset.records().select(Axis(0), indices);
    let targets = dataset.targets().select(Axis(0), indices);
    Dataset::new(records, targets)
}
