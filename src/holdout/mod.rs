pub mod basic_holdout;

pub use basic_holdout::BasicHoldout;

use crate::{data::SpectrumSample, dataset::SpectraData};

/// Defines the methods for a single holdout
pub trait Holdout {
    /// Number of output classes this holdout trains/evaluates.
    fn num_classes(&self) -> usize {
        self.class_indices().len()
    }
    /// Returns the indices of the classes from the `ELEMENTS` constant
    fn class_indices(&self) -> &[usize];
    /// The iteration of the holdout
    fn holdout_number(&self) -> usize;
    /// the value of the random seed for the holdout
    fn random_seed(&self) -> usize;
    /// Returns the training [`SpectraData`] set
    fn train_dataset(&self) -> &SpectraData;
    /// Returns the validation [`SpectraData`] set
    fn validation_dataset(&self) -> &SpectraData;
    /// Returns a tuple of slices of the training and validation [`SpectrumSample`]
    fn split(&self) -> (&[SpectrumSample], &[SpectrumSample]) {
        (
            &self.train_dataset().dataset,
            &self.validation_dataset().dataset,
        )
    }
    /// the total spectra that are in the holdout's training set
    fn training_len(&self) -> usize {
        self.split().0.len()
    }
    /// the total spectra in the holdout's validation set
    fn validation_len(&self) -> usize {
        self.split().1.len()
    }
}

pub trait HoldoutGenerator {
    type HoldoutType: Holdout;

    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType>;
}

pub struct RandomHoldoutConfig {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
}

pub struct StratifiedHoldoutConfig {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
    pub mini_validation_positives_per_class: usize
}