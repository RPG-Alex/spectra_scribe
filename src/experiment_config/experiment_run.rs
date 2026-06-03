use crate::experiment_config::{evaluation::EvaluationConfig, feature::FeatureConfig, loss::LossConfig, mlp_model::MlpModelConfig, optimizer::OptimizerConfig, run::RunConfig};

pub struct ExperimentRunConfig<P> {
    pub run: RunConfig,
    pub features: FeatureConfig,
    pub protocol: P,
    pub model: MlpModelConfig,
    pub optimizer: OptimizerConfig,
    pub loss: LossConfig,
    pub evaluation: EvaluationConfig,
}