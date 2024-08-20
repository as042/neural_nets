#![allow(unused_imports)]
use neural_nets::prelude::*;
use rand::Rng;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(1))
        .add_layer(Layer::new().add_neurons(1))
        .build();

    panic!("{:#?}", net);
}
