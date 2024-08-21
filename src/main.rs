#![allow(unused_imports)]
use std::f64::consts::E;

use neural_nets::prelude::*;
use rand::Rng;
use autodiff::*;

fn main() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(2))
        .add_layer(Layer::new().add_neurons(2))
        .build();

    net.randomize_params(Some(3));

    let settings = &RunSettings::new(
        vec![0.2, 0.15], 
        ActivationFunction::Sigmoid
    );
    let desired_output = vec![0.5, 0.7];

    net.run(settings);
    let init_cost = net.total_cost(&desired_output.iter().map(|f| F::new(*f, 0.0)).collect());

    for _ in 0..10000 {
        println!("{:?}", net.train(settings, 
            &vec![0.5, 0.7], 
            0.1).cost()
        );
    }

    println!("{init_cost}");

    // println!("{:#?}", net);

    // panic!("{}, {:?}", tnn(&[F::new(0.1, 0.0), F::new(-0.1, 0.0)]), grad(tnn, &[0.203, 0.106]));
}

// fn tnn(x: &[FT<f64>]) -> FT<f64> {
//     let w = x[0];
//     let b = x[1];
//     let i = F::new(0.5, 0.0);
//     let o = F::new(0.9, 0.0);

//     let z = w * i + b;

//     let a = F::new(1.0, 0.0) / (F::new(1.0, 0.0) + F::new(E, 0.0).powf(z * F::new(-1.0, 0.0)));
//     let diff: F<f64, f64> = a - o;
//     let cost = diff.powf(F::new(2.0, 0.0));

//     cost
// }