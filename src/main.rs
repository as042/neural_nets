#![allow(unused_imports)]
use neural_nets::prelude::*;
use rand::Rng;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(2))
        .add_layer(Layer::new().add_neurons(2))
        .build();

    println!("{:#?}", net);

    net.genetic_train(test_util);

    println!("{:#?}", net);
}

fn test_util(input: &Vec<f64>) -> Vec<f64> {
    if input.is_empty() {
        let mut rng = rand::thread_rng();
        
        return vec![0.0; 2].into_iter().map(|_| rng.gen::<f64>()).collect();
    }

    let score = -(input.iter().fold(0.0, |acc, x| acc + x)).abs();

    vec![score]
}