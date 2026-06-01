use crate::{
    error::SpectraError,
    experiment_config::{run_experiment, standard::StandardConfig},
};

fn experiment_6_config() -> StandardConfig {
    StandardConfig {
        // Experiment variable changed. modifying weight range
        learning_rate: 1.0,
        experiment_num: 6,
        ..StandardConfig::default()
    }
}

pub fn run() -> Result<(), SpectraError> {
    run_experiment(experiment_6_config())
}
