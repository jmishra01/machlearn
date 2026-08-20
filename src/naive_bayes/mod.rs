//! Probabilistic classifiers and their fitted models.

mod bernoulli_nb;
mod common;
mod gaussian_nb;
mod multinomial_nb;

pub use bernoulli_nb::{BernoulliNaiveBayes, FittedBernoulliNaiveBayes};
pub use gaussian_nb::{FittedGaussianNaiveBayes, GaussianNaiveBayes};
pub use multinomial_nb::{FittedMultinomialNaiveBayes, MultinomialNaiveBayes};
