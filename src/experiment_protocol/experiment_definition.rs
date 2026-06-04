use crate::{
    experiment_protocol::{
        evaluation::EvaluationConfig, feature::FeatureConfig, loss::LossConfig, mlp_model::MlpModelConfig, protocol_config::ProtocolConfig, run::RunConfig
    },
    training::TrainingConfig,
};

pub struct ExperimentDefinition {
    pub run: RunConfig,
    pub features: FeatureConfig,
    pub protocol: ProtocolConfig,
    pub model: MlpModelConfig,
    pub training: TrainingConfig,
    pub loss: LossConfig,
    pub evaluation: EvaluationConfig,
}



