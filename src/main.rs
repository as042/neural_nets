#![allow(unused_imports)]
use neural_nets::prelude::*;
use rand::Rng;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(1))
        .add_layer(Layer::new().add_neurons(1))
        .build();

    net.genetic_train(Identity, 100);

    panic!("{:#?}", net);
}

#[derive(Clone, Copy, Default)]
pub struct Identity;

impl Util for Identity {
    fn gen_input() -> Vec<f64> {
        let mut rng = rand::thread_rng();
        
        vec![0.0; 1].into_iter().map(|_| rng.gen::<f64>()).collect()
    }

    fn evaluate(input: Option<&Vec<f64>>, output: &Vec<f64>) -> f64 {
        let score = -(input.unwrap()[0] - output[0]).abs();

        output[0]
    }
}