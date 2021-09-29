use anyhow::Result;
use tch::{nn, Device, Tensor};
use tch::nn::{Module, OptimizerConfig};
use crate::settings::Settings;

pub fn mlp(_num_layers: i64, input_dim: i64, hidden_dim: i64, output_dim: i64, vs: &nn::Path) -> impl Module {

    // note that we still need to dynamically modify the number of layers, we we've hardcoded 2 mlp layers
    // println!("num_layers:{:?},input_dim:{:?},hidden_dim:{:?},output_dim:{:?}",num_layers,input_dim,hidden_dim,output_dim );
    nn::seq()
        .add(nn::linear(vs, input_dim, hidden_dim, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, hidden_dim, output_dim, Default::default()))
        .add_fn(|xs| xs.relu())
}

pub fn run(config: &Settings) -> Result<()> {


    ////// DATA SETUP

    let num_classes = 2;
    let input_dim = 196;

    let x_concat_file = std::env::current_dir()?.join(format!("{}","dataset/x_concat.pt"));
    let x_labels_file = std::env::current_dir()?.join(format!("{}","dataset/x_labels.pt"));
    let y_concat_file = std::env::current_dir()?.join(format!("{}","dataset/y_concat.pt"));
    let y_labels_file = std::env::current_dir()?.join(format!("{}","dataset/y_labels.pt"));

    println!("loading file from: {:?}", x_concat_file);
    println!("loading file from: {:?}", x_labels_file);
    println!("loading file from: {:?}", y_concat_file);
    println!("loading file from: {:?}", y_labels_file);

    let x_concat = Tensor::load(&x_concat_file).unwrap(); 
    let x_labels = Tensor::load(&x_labels_file).unwrap(); 
    let y_concat = Tensor::load(&y_concat_file).unwrap(); 
    let y_labels = Tensor::load(&y_labels_file).unwrap(); 

    ////// COMPUTE BLOCK SETUP

    // device
    let vs = nn::VarStore::new(Device::cuda_if_available()); // Returns a GPU device if available, else default to CPU.

    // set up seeds and gpu device
    tch::manual_seed(config.data.seed);

    // mlp structure
    let mlp_model = mlp(
        config.nn.num_layers,
        input_dim,
        config.nn.hidden_dim,
        num_classes as i64,
        &vs.root()
    );
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    // train
    for epoch in 1..config.iterations.epochs {
        let loss = mlp_model
            .forward(&x_concat)
            .cross_entropy_for_logits(&x_labels);
        opt.backward_step(&loss);

        let test_accuracy = mlp_model.forward(&y_concat).accuracy_for_logits(&y_labels);
        println!( "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",epoch,f64::from(&loss),100. * f64::from(&test_accuracy),);
    }

    ////// STORE WEIGHTS

    // let filename = std::env::current_dir()?.join(format!("{}",config.store.weights)); // persistm model to file
    // vs.save(&filename);
    // println!("weights tensor saved to: {:?}",filename);
    
    Ok(())
}