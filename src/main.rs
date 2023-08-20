#![allow(unused_imports)]
use neural_nets::prelude::*;

fn main() {
    let net = Network::new()
        .add_layer(Layer::new().add_neurons(5))
        .add_layers(3, Layer::new().add_neurons(10))
        .build();
}