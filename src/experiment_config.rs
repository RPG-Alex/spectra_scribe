use crate::{dataset::SpectraData, holdout::Holdout};

/// Defines the methods for setting up an experiment
pub trait ExperimentConfig {
    type HoldoutType: Holdout;
    /// the Total number of holdouts
    fn number_of_holdouts(&self) -> usize;
    /// The value of set for the random seed of the experiment
    fn random_seed(&self) -> u64;
    /// the percent of the data split into the training set
    fn training_size(&self) -> f32;
    /// the percent of the data split into the validation set
    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }
    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType>;
}
