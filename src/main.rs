#![allow(unused_imports)]
use neural_nets::prelude::*;
use rand::Rng;
use autodiff::*;

fn main() {
    // diff(&fge, 3.0);

    // println!("{}, {}, {:?}", fge(3.0.into()), diff(&fge, 3.0), grad(&eeg, &[10.0, -30.0]));

    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(2))
        .add_layer(Layer::new().add_neurons(2))
        .build();

    net.randomize_params(Some(0));

    panic!("{:#?}", net.train(&vec![0.2, 0.15], &vec![0.5, 0.7]));
}

// fn fge(x: FT<f64>) -> FT<f64> {
//     x.powf(2.0.into())
// }

// fn eeg(x: &[FT<f64>]) -> FT<f64> {
//     x[0].powf((1.0 / x[1].to_f64().unwrap()).into()).into()
// }