use crate::{
    error::SpectraError,
    experiment::{
        ClassWeighting, EvaluationConfig, ExperimentConfig, FeatureConfig, LossConfig,
        MlpModelConfig, RunConfig, StratifiedRetryProtocol, TrainingParams, run_experiment,
    },
};

pub fn run() -> Result<(), SpectraError> {
    let config = ExperimentConfig {
        run: RunConfig {
            experiment_num: 1,
            name: "stratified-retry-baseline-mlp".to_string(),
        },
        features: FeatureConfig { bin_size: 1000 },
        protocol: StratifiedRetryProtocol {
            number_of_holdouts: 5,
            random_seed: 42,
            training_size: 0.8,
            retries_per_holdout: 100,
        },
        model: MlpModelConfig {
            hidden_size: 100,
            dropout: 0.5,
        },
        training: TrainingParams {
            epochs: 10,
            batch_size: 64,
            workers: 4,
            learning_rate: 1.0e-4,
        },
        loss: LossConfig {
            class_weighting: ClassWeighting::InverseFrequency { clamp: (0.1, 10.0) },
        },
        evaluation: EvaluationConfig {
            thresholds: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        },
    };
    run_experiment(config)
}
