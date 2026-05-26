use crate::{
    error::SpectraError,
    experiment_config::{run_experiment, standard::StandardConfig},
};

pub fn run() -> Result<(), SpectraError> {
    run_experiment(StandardConfig::default())
}
