//! Integration tests for train/test splitting.

use approx::assert_abs_diff_eq;
use machlearn::{Dataset, MlError, SplitOptions, train_test_split};
use ndarray::{Axis, array};

fn dataset() -> Dataset<u8> {
    Dataset::new(
        array![[0.0], [1.0], [2.0], [3.0], [4.0]],
        array![0, 1, 2, 3, 4],
    )
    .unwrap()
}

#[test]
fn split_without_shuffle_preserves_order() {
    let (train, test) =
        train_test_split(&dataset(), SplitOptions::new(0.4).with_shuffle(false)).unwrap();

    assert_eq!(train.targets(), array![0, 1, 2].view());
    assert_eq!(test.targets(), array![3, 4].view());
}

#[test]
fn seeded_splits_are_reproducible_and_disjoint() {
    let options = SplitOptions::new(0.4).with_seed(17);
    let (first_train, first_test) = train_test_split(&dataset(), options).unwrap();
    let (second_train, second_test) = train_test_split(&dataset(), options).unwrap();

    assert_eq!(first_train, second_train);
    assert_eq!(first_test, second_test);

    let mut labels: Vec<_> = first_train
        .targets()
        .iter()
        .chain(first_test.targets().iter())
        .copied()
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, vec![0, 1, 2, 3, 4]);
}

#[test]
fn applies_the_same_indices_to_records_and_targets() {
    let (train, test) = train_test_split(&dataset(), SplitOptions::new(0.4)).unwrap();

    for part in [&train, &test] {
        for (row, target) in part.records().axis_iter(Axis(0)).zip(part.targets()) {
            assert_abs_diff_eq!(row[0], f64::from(*target));
        }
    }
}

#[test]
fn rejects_invalid_fractions() {
    for fraction in [-1.0, 0.0, 1.0, 2.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            train_test_split(&dataset(), SplitOptions::new(fraction)),
            Err(MlError::InvalidTestFraction(_))
        ));
    }
}

#[test]
fn rejects_single_sample_datasets() {
    let one = Dataset::new(array![[1.0]], array![1.0]).unwrap();
    assert_eq!(
        train_test_split(&one, SplitOptions::default()).unwrap_err(),
        MlError::InsufficientSamples {
            required: 2,
            actual: 1,
        }
    );
}
