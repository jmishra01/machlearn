//! Internal numerical-solver boundary.
//!
//! Backend vocabulary types stay inside this module so public estimators can
//! retain stable `ndarray`-based APIs if the implementation changes later.

// ndarray views are lightweight descriptors; accepting them by value avoids
// requiring callers to borrow temporary views.
#![allow(clippy::needless_pass_by_value)]

use faer::{Mat, linalg::solvers::SolveLstsq};
use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::core::{MlError, Result, validate_features};

/// Solves an overdetermined, full-column-rank least-squares system with
/// column-pivoted QR decomposition.
#[allow(dead_code)] // Used by the ordinary least-squares milestone that follows.
pub(crate) fn solve_least_squares(
    design: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Result<Array1<f64>> {
    validate_features(design)?;
    if design.nrows() != targets.len() {
        return Err(MlError::MismatchedSampleCount {
            feature_rows: design.nrows(),
            target_count: targets.len(),
        });
    }
    if design.nrows() < design.ncols() {
        return Err(MlError::InsufficientSamples {
            required: design.ncols(),
            actual: design.nrows(),
        });
    }
    for (index, &target) in targets.iter().enumerate() {
        if !target.is_finite() {
            return Err(MlError::NonFiniteActualTarget { index });
        }
    }

    let matrix = Mat::from_fn(design.nrows(), design.ncols(), |row, column| {
        design[[row, column]]
    });
    let decomposition = matrix.col_piv_qr();
    let rank = numerical_rank(decomposition.R(), design.nrows(), design.ncols());
    if rank < design.ncols() {
        return Err(MlError::RankDeficientDesign {
            rank,
            feature_count: design.ncols(),
        });
    }

    let right_hand_side = Mat::from_fn(targets.len(), 1, |row, _column| targets[row]);
    let solution = decomposition.solve_lstsq(right_hand_side.as_ref());
    let coefficients = Array1::from_iter((0..design.ncols()).map(|row| solution[(row, 0)]));
    if let Some((index, _coefficient)) = coefficients
        .iter()
        .enumerate()
        .find(|(_index, coefficient)| !coefficient.is_finite())
    {
        return Err(MlError::NonFiniteSolverOutput { index });
    }
    Ok(coefficients)
}

fn numerical_rank(
    upper_triangular: faer::MatRef<'_, f64>,
    row_count: usize,
    column_count: usize,
) -> usize {
    let diagonal_scale = (0..column_count)
        .map(|index| upper_triangular[(index, index)].abs())
        .fold(0.0_f64, f64::max);
    #[allow(clippy::cast_precision_loss)]
    let dimension = row_count.max(column_count) as f64;
    let tolerance = f64::EPSILON * dimension * diagonal_scale;
    (0..column_count)
        .filter(|&index| upper_triangular[(index, index)].abs() > tolerance)
        .count()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::solve_least_squares;
    use crate::core::MlError;

    #[test]
    fn solves_an_overdetermined_reference_system() {
        let design = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let targets = array![1.0, 3.0, 5.0, 7.0];

        let coefficients = solve_least_squares(design.view(), targets.view()).unwrap();

        assert_abs_diff_eq!(coefficients[0], 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(coefficients[1], 2.0, epsilon = 1.0e-12);
    }

    #[test]
    fn solves_a_square_full_rank_system() {
        let design = array![[2.0, 1.0], [1.0, -1.0]];
        let targets = array![5.0, 1.0];

        let coefficients = solve_least_squares(design.view(), targets.view()).unwrap();

        assert_abs_diff_eq!(coefficients[0], 2.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(coefficients[1], 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn rejects_rank_deficient_designs() {
        let design = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let targets = array![1.0, 2.0, 3.0];

        assert_eq!(
            solve_least_squares(design.view(), targets.view()).unwrap_err(),
            MlError::RankDeficientDesign {
                rank: 1,
                feature_count: 2,
            }
        );
    }

    #[test]
    fn rejects_underdetermined_and_mismatched_systems() {
        let underdetermined = array![[1.0, 2.0]];
        assert_eq!(
            solve_least_squares(underdetermined.view(), array![1.0].view()).unwrap_err(),
            MlError::InsufficientSamples {
                required: 2,
                actual: 1,
            }
        );

        let mismatched = array![[1.0], [2.0]];
        assert_eq!(
            solve_least_squares(mismatched.view(), array![1.0].view()).unwrap_err(),
            MlError::MismatchedSampleCount {
                feature_rows: 2,
                target_count: 1,
            }
        );
    }

    #[test]
    fn rejects_non_finite_targets() {
        let design = array![[1.0], [2.0]];

        assert_eq!(
            solve_least_squares(design.view(), array![1.0, f64::NAN].view()).unwrap_err(),
            MlError::NonFiniteActualTarget { index: 1 }
        );
    }
}
