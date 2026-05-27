use crate::{
    error::SpectraError,
    experiment_config::{run_experiment, standard::StandardConfig},
};

fn experiment_4_config() -> StandardConfig {
    StandardConfig {
        // Experiment variable changed. No weights added
        batch_size: 1,
        experiment_num: 4,
        ..StandardConfig::default()
    }
}

pub fn run() -> Result<(), SpectraError> {
    run_experiment(experiment_4_config())
}
