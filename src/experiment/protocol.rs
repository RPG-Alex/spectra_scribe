use rand::{rngs::ChaCha8Rng, seq::SliceRandom, SeedableRng};

use crate::{
    data::SpectrumSample,
    dataset::{observed_class_indices, SpectraData},
    holdout::{BasicHoldout, Holdout},
};

/// Defines the experiment holdout split
pub trait ExperimentProtocol {
    type HoldoutType: Holdout;

    fn number_of_holdouts(&self) -> usize;
    fn random_seed(&self) -> u64;
    fn training_size(&self) -> f32;
    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }
    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType>;
}

/// Random Split for training/validation
pub struct RandomSplitProtocol {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
}

impl ExperimentProtocol for RandomSplitProtocol {
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
        let class_indices = observed_class_indices(dataset.samples());
        (0..self.number_of_holdouts).map(|holdout_number| {
            let holdout_seed = self.random_seed + holdout_number as u64;
            make_random_holdout(
                dataset,
                &class_indices,
                holdout_number,
                holdout_seed,
                self.training_size
            )
        }).collect()
    }
}

/// Stratified approach to random split. Yields best possible split for class distribution
pub struct StratifiedRetryProtocol {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
    pub retries_per_holdout: usize,
}

impl ExperimentProtocol for StratifiedRetryProtocol {
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
        let class_indices= observed_class_indices(dataset.samples());
        let attempts = self.retries_per_holdout.max(1);
        (0..self.number_of_holdouts) .map(|holdout_number| {
            let mut best: Option<(f32, Vec<SpectrumSample>, Vec<SpectrumSample>, u64)> = None;
            for attempt in 0..attempts {
                let seed = self.random_seed + holdout_number as u64 *10_000 + attempt as u64;
                let (train, validation) = random_split(dataset, seed, self.training_size);
                let score = split_score(&train, &validation, &class_indices, self.training_size);
                match &best {
                    Some((best_score, _,_,_)) if *best_score <= score => {},
                    _ => best = Some((score, train, validation, seed)),
                }
            }
            let (_score, train, validation, seed) = best.expect("Need at least one split attempt");
            BasicHoldout::new(
                SpectraData::from_samples(train, dataset.bin_size()),
                SpectraData::from_samples(validation, dataset.bin_size()),
                class_indices.clone(),
                holdout_number,
                seed as usize,
            )
        })
        .collect()
    }
}

fn make_random_holdout(
    dataset: &SpectraData,
    class_indices: &[usize],
    holdout_number: usize,
    holdout_seed: u64,
    training_size: f32,
) -> BasicHoldout {
    let (train, validation) = random_split(dataset, holdout_seed, training_size);
    BasicHoldout::new(
        SpectraData::from_samples(train, dataset.bin_size()),
        SpectraData::from_samples(validation, dataset.bin_size()),
        class_indices.to_vec(),
        holdout_number,
        holdout_seed as usize
    )
}

fn random_split(
    dataset: &SpectraData,
    seed: u64,
    training_size: f32,
) -> (Vec<SpectrumSample>, Vec<SpectrumSample>) {
    let mut samples = dataset.samples().to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    samples.shuffle(&mut rng);

    let split_index = (samples.len() as f32 * training_size) as usize;
    let validation = samples.split_off(split_index);
    let train = samples;

    (train, validation)
}

fn split_score(
    train: &[SpectrumSample],
    validation: &[SpectrumSample],
    class_indices: &[usize],
    training_size: f32,
) -> f32 {
    let expected_validation_fraction = 1.0 - training_size;
    let mut score = 0.0;
    for &class_index in class_indices {
        let train_positive =  train.iter().filter(|sample| sample.element_present[class_index]).count();
        let validation_positive = validation.iter().filter(|sample| sample.element_present[class_index]).count();
        let total_positive = train_positive + validation_positive;
        if total_positive == 0 {
            continue;
        }
        if train_positive == 0 {
            score += 1_000.0;
        }
        if validation_positive == 0 && total_positive > 1 {
            score += 1_000.0;
        }
        let actual_validation_fraction = validation_positive as f32 / total_positive as f32;
        score += (actual_validation_fraction - expected_validation_fraction).abs();
    }
    score
}