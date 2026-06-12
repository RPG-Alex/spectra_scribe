//! CLI entry point for running experiments

use crate::{error::SpectraError, experiments::experiment1};

fn main() -> Result<(), SpectraError> {
    experiment1::run()
}
