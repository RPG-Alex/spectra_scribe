use crate::experiment_protocol::split_strategy::SplitStrategy;

pub struct ProtocolConfig {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
    pub split_strategy: SplitStrategy,
}