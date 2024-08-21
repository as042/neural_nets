#![allow(unused_imports)]
use std::f64::consts::E;

use neural_nets::prelude::*;
use rand::Rng;
use autodiff::*;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(1))
        .add_layer(Layer::new().add_neurons(3))
        .add_layer(Layer::new().add_neurons(5))
        .add_layer(Layer::new().add_neurons(9))
        .add_layer(Layer::new().add_neurons(3))
        .build();

    net.randomize_params(None);

    let settings = &RunSettings::new(
        vec![0.2], 
        ActivationFunction::Tanh,
        true
    );
    let desired_output = vec![0.5, 0.7, 0.56];

    net.run(settings);
    let init_cost = net.total_cost(&desired_output.iter().map(|f| F::new(*f, 0.0)).collect());

    for _ in 0..1000 {
        println!("{:?}", net.train(settings, 
            &vec![0.5, 0.7, 0.56], 
            0.1).cost()
        );
    }

    println!("{init_cost}");
}