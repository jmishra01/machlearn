mod classification;
mod regression;

pub use classification::{ConfusionMatrix, accuracy_score, confusion_matrix};
pub use regression::{mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error};
