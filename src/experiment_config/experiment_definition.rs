use crate::{experiment_config::{evaluation::EvaluationConfig, feature::FeatureConfig, loss::LossConfig, mlp_model::MlpModelConfig, run::RunConfig}, training::TrainingConfig};

pub struct ExperimentDefinition {
    pub run: RunConfig,
    pub features: FeatureConfig,
    pub protocol: ProtocolConfig,
    pub model: MlpModelConfig,
    pub training: TrainingConfig,
    pub loss: LossConfig,
    pub evaluation: EvaluationConfig,
}





pub struct ProtocolConfig {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
    pub split_strategy: SplitStrategy
}

pub enum SplitStrategy {
    Random,
    StratifiedMultiLabel
}





pub struct ClassDistribution {
    pub class_index: usize,
    pub train_positive: usize,
    pub validation_positive: usize,
    pub total_positive: usize,
}