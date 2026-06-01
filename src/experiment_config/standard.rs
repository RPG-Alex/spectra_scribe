use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};

use crate::{
    dataset::SpectraData, experiment_config::ExperimentConfig, experiments::observed_class_indices,
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

impl ExperimentConfig for StandardConfig {
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
    fn bin_size(&self) -> usize {
        self.bin_size
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

    fn epochs(&self) -> usize {
        self.epochs
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn num_workers(&self) -> usize {
        self.workers
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn weight_range(&self) -> Option<(f32, f32)> {
        self.weight_range
    }

    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }

    fn threshold(&self) -> f64 {
        0.5
    }
    fn experiment_num(&self) -> usize {
        self.experiment_num
    }

    fn dropout(&self) -> f64 {
        self.dropout
    }
    fn experiment_details(&self) -> String {
        let mut out: String = String::new();
        out.push_str(&format!("Experiment Number: {}\n", self.experiment_num));
        out.push_str(&format!(
            "Number of Holdouts: {}\n",
            self.number_of_holdouts
        ));
        out.push_str(&format!("Random seed: {}\n", self.random_seed));
        out.push_str(&format!("Training Size: {}\n", self.training_size));
        out.push_str(&format!("Number of Epochs: {}\n", self.epochs));
        out.push_str(&format!("Batch Size: {}\n", self.batch_size));
        out.push_str(&format!("Number of Workers: {}\n", self.workers));
        out.push_str(&format!("Learning Rate: {}\n", self.learning_rate));
        out.push_str(&format!("Hidden Layer Size: {}\n", self.hidden_size));
        out.push_str(&format!("Bin Size: {}\n", self.bin_size));
        match self.weight_range {
            Some((min, max)) => {
                out.push_str(&format!("Weight Range: {}-{}\n", min, max));
            }
            None => out.push_str("No weights used\n"),
        };
        out.push_str(&format!("Dropout: {}\n", self.dropout));

        out
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
