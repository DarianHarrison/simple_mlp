#![allow(warnings)]

use anyhow::Result;
use rhagle::gnn::{ mlp, prediction};
use rhagle::settings;

fn main() -> Result<()> {

    // import config
    let config = settings::Settings::new().expect("load config");

    // command line arguments
    let args: Vec<String> = std::env::args().collect();
    let model = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };

    // compute block to run with metadata(config)
    match model {
        None => mlp::run(&config),
        Some("mlp") => mlp::run(&config),
        Some("predict") => prediction::run(&config),
        Some(_) => mlp::run(&config),
    }
}