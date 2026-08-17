//! Integration tests for K-fold partitioning.

use machlearn::{KFold, MlError};

#[test]
fn creates_balanced_ordered_folds() {
    let folds = KFold::new(3).unwrap().split(10).unwrap();

    assert_eq!(folds.len(), 3);
    assert_eq!(folds[0].test_indices(), &[0, 1, 2, 3]);
    assert_eq!(folds[1].test_indices(), &[4, 5, 6]);
    assert_eq!(folds[2].test_indices(), &[7, 8, 9]);
    assert_eq!(
        folds
            .iter()
            .map(machlearn::Fold::test_size)
            .collect::<Vec<_>>(),
        vec![4, 3, 3]
    );
}

#[test]
fn test_partitions_cover_every_sample_exactly_once() {
    let sample_count = 17;
    let folds = KFold::new(5).unwrap().split(sample_count).unwrap();
    let mut test_indices: Vec<_> = folds
        .iter()
        .flat_map(|fold| fold.test_indices().iter().copied())
        .collect();
    test_indices.sort_unstable();

    assert_eq!(test_indices, (0..sample_count).collect::<Vec<_>>());
}

#[test]
fn train_and_test_indices_are_disjoint_complements() {
    let sample_count = 11;
    let folds = KFold::new(4).unwrap().split(sample_count).unwrap();

    for fold in folds {
        assert_eq!(fold.train_size() + fold.test_size(), sample_count);
        let mut combined = fold.train_indices().to_vec();
        combined.extend_from_slice(fold.test_indices());
        combined.sort_unstable();
        assert_eq!(combined, (0..sample_count).collect::<Vec<_>>());
    }
}

#[test]
fn seeded_shuffling_is_reproducible() {
    let splitter = KFold::new(4).unwrap().with_shuffle(true).with_seed(2026);
    let first = splitter.split(20).unwrap();
    let second = splitter.split(20).unwrap();

    assert_eq!(first, second);
    assert_ne!(first, KFold::new(4).unwrap().split(20).unwrap());
}

#[test]
fn supports_leave_one_out_sized_folds() {
    let folds = KFold::new(4).unwrap().split(4).unwrap();

    assert!(folds.iter().all(|fold| fold.test_size() == 1));
    assert!(folds.iter().all(|fold| fold.train_size() == 3));
}

#[test]
fn rejects_invalid_fold_counts_and_insufficient_samples() {
    assert_eq!(
        KFold::new(1).unwrap_err(),
        MlError::InvalidFoldCount { n_splits: 1 }
    );
    assert_eq!(
        KFold::new(5).unwrap().split(4).unwrap_err(),
        MlError::InsufficientSamples {
            required: 5,
            actual: 4,
        }
    );
}

#[cfg(feature = "serde")]
#[test]
fn folds_round_trip_through_serde() {
    let folds = KFold::new(3).unwrap().split(8).unwrap();
    let json = serde_json::to_string(&folds).unwrap();
    let restored: Vec<machlearn::Fold> = serde_json::from_str(&json).unwrap();

    assert_eq!(folds, restored);
}
