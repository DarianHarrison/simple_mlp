#![allow(warnings)]
use anyhow::Result;
use tch::{nn::Module, Device, Tensor};
use crate::settings::Settings;
use crate::gnn::mlp::{ mlp };

pub fn run(config: &Settings) -> Result<()> {

    ////// LOAD MODELS AND WEIGHTS FOR PREDICTION
    let mut vs = tch::nn::VarStore::new(Device::cuda_if_available());

    // need to override number of classes
    let num_classes:i64 = 2;

    // override input dimensions
    let input_dim:i64 = 196; // will be dynamically set depending on data

    let y_concat_file_unlabeled_file = std::env::current_dir()?.join(format!("{}","dataset/y_concat_unlabeled.pt"));
    let y_concat_unlabeled = Tensor::load(&y_concat_file_unlabeled_file).unwrap(); 
    //println!("loading file from: {:?}", y_concat_file_unlabeled);

    // create the net (and so create the variables) before calling vs.load, their weights will actually be loaded from the file
    let mlp_model = mlp(
        config.nn.num_layers,
        input_dim,
        config.nn.hidden_dim,
        num_classes,
        &vs.root()
    );

    // load model weights to mlp
    let model_filename = std::env::current_dir()?.join(format!("{}", config.store.weights));
    vs.load(&model_filename);
    println!("loaded model from: {:?}", &model_filename);

    ////// NN SETUP

    let prediction = mlp_model
        .forward(&y_concat_unlabeled)
        .softmax(-1, tch::Kind::Float); // Convert to probability.

    let pred: Vec<f64> = Vec::<f64>::from(&prediction);
    println!("{:?}", &pred);

    Ok(())
}