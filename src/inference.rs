use crate::{
    data::{SpectraScribeBatcher, SpectrumSample},
    training::TrainingConfig,
};

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    items: Vec<SpectrumSample>,
) -> Tensor<B, 2> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(device).load_record(record);

    let batcher = SpectraScribeBatcher::new(config.class_indices.clone(), config.model.bin_size());
    let batch = batcher.batch(items, device);
    model.forward(batch.spectra)
}
