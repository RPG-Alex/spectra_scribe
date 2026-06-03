use crate::{dataset::SpectraData, holdout::Holdout};
pub mod runner;
pub mod standard;
pub mod evaluation;
pub mod experiment_definition;
pub mod experiment_run;
pub mod feature;
pub mod loss;
pub mod mlp_model;
pub mod optimizer;
pub mod run;

pub use runner::run_experiment;

/// Defines the methods for setting up an experiment
pub trait ExperimentProtocol {
    type HoldoutType: Holdout;
    /// How many independent train/validation splits to run for this experiment.
    fn number_of_holdouts(&self) -> usize;

    /// Base seed used to make dataset shuffling and holdout generation reproducible.
    fn random_seed(&self) -> u64;

    /// Fraction of the dataset used for training in each holdout.
    fn training_size(&self) -> f32;

    /// Fraction of the dataset held back for validation in each holdout.
    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }

    /// Builds the concrete train/validation holdouts this experiment will run.
    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType>;
}
