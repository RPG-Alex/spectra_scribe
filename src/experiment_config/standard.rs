use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};

use crate::{
    dataset::SpectraData, experiment_config::ExperimentProtocol, experiments::observed_class_indices,
    holdout::BasicHoldout,
};

pub struct StandardConfig {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub workers: usize,
    pub learning_rate: f64,
    pub hidden_size: usize,
    pub bin_size: usize,
    pub weight_range: Option<(f32, f32)>,
    pub experiment_num: usize,
    pub dropout: f64,
}

impl ExperimentProtocol for StandardConfig {
    type HoldoutType = BasicHoldout;

    fn number_of_holdouts(&self) -> usize {
        self.number_of_holdouts
    }

    fn random_seed(&self) -> u64 {
        self.random_seed
    }

    fn training_size(&self) -> f32 {
        self.training_size
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

impl Default for StandardConfig {
    fn default() -> Self {
        Self {
            number_of_holdouts: 1,
            random_seed: 42,
            training_size: 0.8,
            epochs: 10,
            batch_size: 256,
            workers: 4,
            learning_rate: 1.0e-4,
            hidden_size: 100,
            bin_size: 1000,
            weight_range: Some((0.1, 10.0)),
            experiment_num: 1,
            dropout: 0.5,
        }
    }
}
