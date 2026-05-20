use std::fs::File;

use burn::backend::{Autodiff, Wgpu};
use csv::Writer;

use crate::{
    data::{ELEMENT_COUNT, ELEMENTS},
    dataset::SpectraData,
    error::SpectraError,
    experiments::observed_class_indices,
    inference::create_confusion_matrices,
    model::ModelConfig,
    training::TrainingConfig,
};

pub fn run() -> Result<(), SpectraError> {
    // Training values
    let epoch: usize = 10;
    let batch: usize = 64;
    let workers: usize = 4;
    let seed: u64 = 42;
    let rate: f64 = 1.0e-4;

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./experiments/experiment1";

    println!("Loading spectra.");
    let dataset = SpectraData::new()?;
    println!("Finished loading spectra");

    let model_config = ModelConfig::new(ELEMENT_COUNT, 100)
        .with_class_weights(Some(dataset.class_weights.clone()));

    let train_dataset = dataset.train(seed);
    let validation_dataset = dataset.test(seed);
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        &train_dataset,
        &validation_dataset,
        TrainingConfig::new_with_values(model_config, epoch, batch, workers, seed, rate),
        device.clone(),
    );

    let validation_items = validation_dataset.dataset.clone();

    let results =
        crate::inference::infer::<MyBackend>(artifact_dir, &device, validation_items.clone());

    let confusion_matrices = create_confusion_matrices(results, &validation_items, 0.5)?;

    let file = File::create("results/experiment1_results.csv")?;
    let mut wtr = Writer::from_writer(file);

    for matrix in confusion_matrices {
        wtr.serialize(matrix)?;
    }

    wtr.flush()?;
    Ok(())
}
