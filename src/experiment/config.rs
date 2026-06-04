/// Complete definition of the one experiment run. 
#[derive(Clone, Debug)]
pub struct ExperimentConfig<P> {
    pub run: RunConfig,
    pub features: FeatureConfig,
    pub protocol: P,
    pub model: MlpModelConfig,
    pub training: TrainingParams,
    pub loss: LossConfig,
    pub evaluation: EvaluationConfig,
}

/// The run's metadata and name
pub struct RunConfig {
    pub experiment_num: usize,
    pub name: String,
}

/// Model Input presentation/shape. 
pub struct FeatureConfig {
    pub bin_size: usize,
}

/// Network architecture settings
pub struct MlpModelConfig {
    pub hidden_size: usize,
    pub dropout: f64,
}

/// Training settings
pub struct TrainingParams {
    pub epochs: usize,
    pub batch_size: usize,
    pub workers: usize,
    pub learning_rate: f64,
}

/// Loss-function
pub struct LossConfig {
    pub class_weighting: ClassWeighting,
}

/// enum for containing the class weight
pub enum ClassWeighting {
    None,
    InverseFrequency { clamp: (f32, f32)}
}

/// Evaluation settings
pub struct EvaluationConfig {
    pub thresholds: Vec<f64>,
}