//! Integration tests for stratified K-fold partitioning.

use std::collections::BTreeMap;

use machlearn::{MlError, StratifiedKFold};

fn class_counts<'a>(fold_indices: &[usize], targets: &[&'a str]) -> BTreeMap<&'a str, usize> {
    let mut counts = BTreeMap::new();
    for &index in fold_indices {
        *counts.entry(targets[index]).or_default() += 1;
    }
    counts
}

#[test]
fn balances_every_class_and_total_fold_sizes() {
    let targets = ["a", "a", "a", "a", "b", "b", "b", "b", "c"];
    let folds = StratifiedKFold::new(3).unwrap().split(&targets).unwrap();

    let fold_counts: Vec<_> = folds
        .iter()
        .map(|fold| class_counts(fold.test_indices(), &targets))
        .collect();
    for label in ["a", "b", "c"] {
        let counts: Vec<_> = fold_counts
            .iter()
            .map(|counts| counts.get(label).copied().unwrap_or_default())
            .collect();
        assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
    }
    assert_eq!(
        folds
            .iter()
            .map(machlearn::Fold::test_size)
            .collect::<Vec<_>>(),
        vec![3, 3, 3]
    );
}

#[test]
fn test_partitions_cover_every_sample_exactly_once() {
    let targets = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3];
    let folds = StratifiedKFold::new(4).unwrap().split(&targets).unwrap();
    let mut test_indices: Vec<_> = folds
        .iter()
        .flat_map(|fold| fold.test_indices().iter().copied())
        .collect();
    test_indices.sort_unstable();

    assert_eq!(test_indices, (0..targets.len()).collect::<Vec<_>>());
}

#[test]
fn train_and_test_indices_are_disjoint_complements() {
    let targets = [0, 1, 0, 2, 1, 2, 0, 1, 2, 2];
    let folds = StratifiedKFold::new(3).unwrap().split(&targets).unwrap();

    for fold in folds {
        assert_eq!(fold.train_size() + fold.test_size(), targets.len());
        let mut combined = fold.train_indices().to_vec();
        combined.extend_from_slice(fold.test_indices());
        combined.sort_unstable();
        assert_eq!(combined, (0..targets.len()).collect::<Vec<_>>());
    }
}

#[test]
fn supports_classes_smaller_than_the_fold_count() {
    let targets = ["major", "major", "major", "major", "minor"];
    let folds = StratifiedKFold::new(5).unwrap().split(&targets).unwrap();

    assert!(folds.iter().all(|fold| fold.test_size() == 1));
    assert_eq!(
        folds
            .iter()
            .filter(|fold| fold.test_indices().contains(&4))
            .count(),
        1
    );
}

#[test]
fn seeded_shuffling_is_reproducible() {
    let targets = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let splitter = StratifiedKFold::new(3)
        .unwrap()
        .with_shuffle(true)
        .with_seed(2026);
    let first = splitter.split(&targets).unwrap();
    let second = splitter.split(&targets).unwrap();

    assert_eq!(first, second);
    assert_ne!(
        first,
        StratifiedKFold::new(3).unwrap().split(&targets).unwrap()
    );
}

#[test]
fn exposes_configuration() {
    let splitter = StratifiedKFold::new(4)
        .unwrap()
        .with_shuffle(true)
        .with_seed(7);

    assert_eq!(splitter.n_splits(), 4);
    assert!(splitter.shuffle());
    assert_eq!(splitter.seed(), 7);
}

#[test]
fn rejects_invalid_fold_counts_and_insufficient_targets() {
    assert_eq!(
        StratifiedKFold::new(1).unwrap_err(),
        MlError::InvalidFoldCount { n_splits: 1 }
    );
    assert_eq!(
        StratifiedKFold::new(4)
            .unwrap()
            .split(&[0, 1, 1])
            .unwrap_err(),
        MlError::InsufficientSamples {
            required: 4,
            actual: 3,
        }
    );
}
