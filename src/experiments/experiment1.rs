use std::{
    fs::{self, File},
    path::PathBuf,
};

use burn::backend::{Autodiff, Wgpu};
use csv::Writer;
use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};

use crate::{
    dataset::SpectraData,
    error::SpectraError,
    experiment_config::ExperimentConfig,
    experiments::observed_class_indices,
    holdout::{BasicHoldout, Holdout},
    inference::create_confusion_matrices,
    model::ModelConfig,
    training::TrainingConfig,
};

pub struct Experiment1Config {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub workers: usize,
    pub learning_rate: f64,
    pub hidden_size: usize,
    pub bin_size: usize,
}

impl Default for Experiment1Config {
    fn default() -> Self {
        Self {
            number_of_holdouts: 1,
            random_seed: 42,
            training_size: 0.8,
            epochs: 10,
            batch_size: 64,
            workers: 4,
            learning_rate: 1.0e-4,
            hidden_size: 100,
            bin_size: 1000,
        }
    }
}

impl ExperimentConfig for Experiment1Config {
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
                class_weights: dataset.class_weights.clone(),
                bin_size: dataset.bin_size(),
            };

            let validation = SpectraData {
                dataset: samples[split_index..].to_vec(),
                class_weights: dataset.class_weights.clone(),
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
}

pub fn run() -> Result<(), SpectraError> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let experiment_config = Experiment1Config::default();

    let device = burn::backend::wgpu::WgpuDevice::default();

    let experiment_dir = "./experiments/experiment1";
    let results_dir = "./results";

    fs::create_dir_all(experiment_dir)?;
    fs::create_dir_all(results_dir)?;

    println!("Loading spectra.");
    let dataset = SpectraData::new(experiment_config.bin_size())?;
    println!("Finished loading spectra.");

    let holdouts = experiment_config.generate_holdouts(&dataset);

    println!("Generated {} holdout(s).", holdouts.len());

    for holdout in holdouts {
    debug_assert_eq!(
        holdout.num_classes(),
        holdout.class_indices().len(),
    );
        println!(
            "Running holdout {} with seed {}.",
            holdout.holdout_number(),
            holdout.random_seed(),
        );

        println!(
            "Training samples: {}, validation samples: {}.",
            holdout.training_len(),
            holdout.validation_len(),
        );

        let artifact_dir = format!("{experiment_dir}/holdout_{}", holdout.holdout_number(),);

        let class_weights = holdout
            .train_dataset()
            .class_weights_for(holdout.class_indices());

        let model_config = ModelConfig::new(
            holdout.num_classes(),
            experiment_config.hidden_size(),
            experiment_config.bin_size(),
        )
        .with_class_weights(Some(class_weights));

        let training_config = TrainingConfig::new_with_values(
            model_config,
            experiment_config.epochs(),
            experiment_config.batch_size(),
            experiment_config.num_workers(),
            holdout.random_seed() as u64,
            experiment_config.learning_rate(),
            holdout.class_indices().to_vec(),
        );

        crate::training::train_holdout::<MyAutodiffBackend, _>(
            &artifact_dir,
            &holdout,
            training_config,
            device.clone(),
        );

        let validation_items = holdout.validation_dataset().dataset.clone();

        let predictions =
            crate::inference::infer::<MyBackend>(&artifact_dir, &device, validation_items.clone());

        let confusion_matrices = create_confusion_matrices(
            predictions,
            &validation_items,
            holdout.class_indices(),
            experiment_config.threshold(),
        )?;

        let results_path = PathBuf::from(results_dir).join(format!(
            "experiment1_holdout_{}_results.csv",
            holdout.holdout_number(),
        ));

        let file = File::create(results_path)?;
        let mut writer = Writer::from_writer(file);

        for matrix in confusion_matrices {
            writer.serialize(matrix)?;
        }

        writer.flush()?;
    }

    Ok(())
}
