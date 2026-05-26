
use crate::{
    error::SpectraError,
    experiment_config::{run_experiment, standard::StandardConfig},
};

fn experiment_2_config() -> StandardConfig {
    StandardConfig {
        // Experiment variable changed. Increase weight max
        weight_range: (0.05, 25.0),
        experiment_num: 2,
        ..StandardConfig::default()
    }
}

pub fn run() -> Result<(), SpectraError> {
    run_experiment(experiment_2_config())
}
