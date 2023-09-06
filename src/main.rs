#![allow(unused_imports)]
use genetic_optimization::sim::Util;
use neural_nets::prelude::*;
use rand::Rng;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(3))
        .add_layer(Layer::new().add_neurons(3))
        .add_layer(Layer::new().add_neurons(3))
        .build();

    println!("{:#?}", net);

    net.genetic_train(UtilTest);

    println!("{:#?}", net);
}

#[derive(Clone, Copy, Default)]
pub struct UtilTest;

impl Util for UtilTest {
    fn gen_input() -> Vec<f64> {
        let mut rng = rand::thread_rng();
        
        vec![0.0; 3].into_iter().map(|_| rng.gen::<f64>()).collect()
    }

    fn evaluate(_input: Option<&Vec<f64>>, output: &Vec<f64>) -> f64 {
        let score = -(0.21 - output.iter().fold(0.0, |acc, x| acc + x)).abs();

        score
    }
}