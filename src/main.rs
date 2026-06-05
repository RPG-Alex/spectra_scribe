#![recursion_limit = "256"]

use crate::{error::SpectraError, experiments::experiment1};

mod data;
mod dataset;
mod error;
mod evaluation;
mod experiment;
mod experiments;
mod holdout;
mod inference;
mod mcc;
mod model;
mod training;

fn main() -> Result<(), SpectraError> {
    experiment1::run()
}
