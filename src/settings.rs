use config::{Config, ConfigError, File};
use serde::Deserialize;

// These structs mirror the fields in Default.toml
// We can use different data types and even nested structures if needed.

#[derive(Debug, Deserialize, Clone)]
pub struct Data { // Observations
    pub dataset: String,
    pub batch_size: i64,
    pub shuffle: bool,
    pub fold_idx: usize,
    pub degree_as_tag: bool,
    pub seed: i64,
    pub predict_data: String, // one sample of data to prove prediction works
}

#[derive(Debug, Deserialize, Clone)]
pub struct NN { // Compute block space
    pub num_layers: i64,
    pub num_mlp_layers: i64,
    pub hidden_dim: i64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Aggregation { // Entropy capture
    pub graph_pooling_type: String,
    pub neighbor_pooling_type: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Learning { // Adjustments
    pub lr: f64,
    pub learn_eps: bool,
    pub final_dropout: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Iterations { // NN Cycles
    pub epochs: i64,
    pub iters_per_epoch: i64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Torch { // mount the compute instructions to physical device
    pub device: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Store { // for now it refers to the filename where we will store the result of transformations 
    pub features: String,
    pub weights: String,
    pub predictions: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub data: Data,
    pub nn: NN,
    pub aggregation: Aggregation,
    pub learning: Learning,
    pub iterations: Iterations,
    pub torch: Torch,
    pub store: Store,
}

// load the config.
impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let mut s = Config::new();
        s.merge(File::with_name(CONFIG_FILE_PATH))?;
        s.try_into()
    }
}

const CONFIG_FILE_PATH: &str = "./config/Default.toml";