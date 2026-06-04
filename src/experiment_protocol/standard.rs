use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};

use crate::{
    dataset::SpectraData, experiment_protocol::{ExperimentProtocol, evaluation::EvaluationConfig, feature::FeatureConfig, loss::LossConfig, mlp_model::MlpModelConfig, protocol_config::ProtocolConfig, run::RunConfig},
    experiments::observed_class_indices, holdout::BasicHoldout, training::TrainingConfig,
};

pub struct StandardExperiment {
    pub run: RunConfig,
    pub features: FeatureConfig,
    pub protocol: ProtocolConfig,
    pub model: MlpModelConfig,
    pub training: TrainingConfig,
    pub loss: LossConfig,
    pub evaluation: EvaluationConfig,
}

impl ExperimentProtocol for StandardExperiment {
    type HoldoutType = BasicHoldout;

    fn number_of_holdouts(&self) -> usize {
        self.protocol.number_of_holdouts
    }

    fn random_seed(&self) -> u64 {
        self.protocol.random_seed
    }

    fn training_size(&self) -> f32 {
        self.protocol.training_size
    }
    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType> {
        let class_indices = observed_class_indices(&dataset.dataset);

        let mut holdouts = Vec::with_capacity(self.number_of_holdouts());

        for holdout_number in 0..self.number_of_holdouts() {
            let holdout_seed = self.random_seed() + holdout_number as u64;

            let mut samples = dataset.dataset.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(holdout_seed);
            samples.shuffle(&mut rng);

            let split_index = (samples.len() as f32 * self.training_size()) as usize;

            let train = SpectraData {
                dataset: samples[..split_index].to_vec(),
                bin_size: dataset.bin_size(),
            };

            let validation = SpectraData {
                dataset: samples[split_index..].to_vec(),
                bin_size: dataset.bin_size(),
            };

            let holdout = BasicHoldout::new(
                train,
                validation,
                class_indices.clone(),
                holdout_number,
                holdout_seed as usize,
            );

            holdouts.push(holdout);
        }

        holdouts
    }

    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }
}
