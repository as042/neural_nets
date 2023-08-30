#![allow(unused_imports)]
use neural_nets::prelude::*;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(2))
        .add_layer(Layer::new().add_neurons(2))
        .build();

    net.set_neuron_params(0, 0.3, vec![0.1, 0.5]);
    net.set_neuron_params(1, -0.08, vec![-0.09, 2.1]);
    net.run(vec![0.5, 0.1]);

    println!("{:#?}", net);
    println!("{:?}", net.output());
}