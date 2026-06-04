use crate::{
    error::SpectraError,
    experiment_protocol::{run_experiment},
};

pub fn run() -> Result<(), SpectraError> {
    run_experiment()
}
