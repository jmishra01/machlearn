//! Integration tests for deterministic hyperparameter grids.

use machlearn::{MlError, ParameterGrid, ParameterValue};

#[test]
fn expands_a_deterministic_cartesian_product() {
    let grid = ParameterGrid::new()
        .with_parameter("shuffle", [false, true])
        .unwrap()
        .with_parameter("alpha", [0.1, 1.0])
        .unwrap();
    let combinations = grid.combinations().unwrap();

    assert_eq!(grid.len(), 2);
    assert_eq!(grid.combination_count().unwrap(), 4);
    assert_eq!(combinations.len(), 4);
    let actual: Vec<_> = combinations
        .iter()
        .map(|parameters| {
            (
                parameters.get("alpha").unwrap().as_f64().unwrap(),
                parameters.get("shuffle").unwrap().as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        actual,
        vec![(0.1, false), (0.1, true), (1.0, false), (1.0, true)]
    );
}

#[test]
fn an_empty_grid_represents_the_default_configuration() {
    let grid = ParameterGrid::new();
    let combinations = grid.combinations().unwrap();

    assert!(grid.is_empty());
    assert_eq!(grid.combination_count().unwrap(), 1);
    assert_eq!(combinations.len(), 1);
    assert!(combinations[0].is_empty());
}

#[test]
fn preserves_supported_value_types() {
    let grid = ParameterGrid::new()
        .with_parameter(
            "mixed",
            vec![
                ParameterValue::from(-2_i64),
                ParameterValue::from(3_u64),
                ParameterValue::from("fast"),
            ],
        )
        .unwrap();
    let combinations = grid.combinations().unwrap();

    assert_eq!(combinations[0].get("mixed").unwrap().as_i64(), Some(-2));
    assert_eq!(combinations[1].get("mixed").unwrap().as_u64(), Some(3));
    assert_eq!(combinations[2].get("mixed").unwrap().as_str(), Some("fast"));
    assert_eq!(combinations[0].len(), 1);
}

#[test]
fn iterates_parameter_sets_in_sorted_name_order() {
    let grid = ParameterGrid::new()
        .with_parameter("zeta", [1])
        .unwrap()
        .with_parameter("alpha", [2])
        .unwrap();
    let combination = grid.combinations().unwrap().pop().unwrap();

    assert_eq!(
        combination
            .iter()
            .map(|(name, _value)| name)
            .collect::<Vec<_>>(),
        vec!["alpha", "zeta"]
    );
}

#[test]
fn supports_mutable_grid_construction() {
    let mut grid = ParameterGrid::new();
    grid.insert_parameter("neighbors", [3_usize, 5]).unwrap();

    assert_eq!(grid.combination_count().unwrap(), 2);
    assert_eq!(
        grid.combinations().unwrap()[1]
            .get("neighbors")
            .unwrap()
            .as_u64(),
        Some(5)
    );
}

#[test]
fn rejects_invalid_names_duplicates_and_empty_values() {
    assert_eq!(
        ParameterGrid::new()
            .with_parameter("  ", [true])
            .unwrap_err(),
        MlError::InvalidParameterName
    );

    let grid = ParameterGrid::new().with_parameter("alpha", [0.1]).unwrap();
    assert_eq!(
        grid.with_parameter("alpha", [1.0]).unwrap_err(),
        MlError::DuplicateParameter {
            name: "alpha".to_owned(),
        }
    );

    assert_eq!(
        ParameterGrid::new()
            .with_parameter("alpha", Vec::<f64>::new())
            .unwrap_err(),
        MlError::EmptyParameterValues {
            name: "alpha".to_owned(),
        }
    );
}

#[test]
fn rejects_non_finite_float_candidates() {
    assert_eq!(
        ParameterGrid::new()
            .with_parameter("alpha", [0.1, f64::NAN])
            .unwrap_err(),
        MlError::NonFiniteParameterValue {
            name: "alpha".to_owned(),
            value_index: 1,
        }
    );
}

#[cfg(feature = "serde")]
#[test]
fn grids_and_combinations_round_trip_through_serde() {
    let grid = ParameterGrid::new()
        .with_parameter("depth", [2_u32, 4])
        .unwrap()
        .with_parameter("criterion", ["gini", "entropy"])
        .unwrap();
    let json = serde_json::to_string(&grid).unwrap();
    let restored: ParameterGrid = serde_json::from_str(&json).unwrap();

    assert_eq!(grid, restored);
    assert_eq!(
        grid.combinations().unwrap(),
        restored.combinations().unwrap()
    );
}
