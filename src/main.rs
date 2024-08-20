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

    net.randomize_params(Some(0));
    let mut net2 = net.clone();
    net2.run(&vec![0.2.into(), 0.15.into()]);

    panic!("{:#?}, {:?}, {}", net.train(&vec![0.2.into(), 0.15.into()], &vec![0.5.into(), 0.7.into()]), net2.output(), net2.total_cost(&vec![0.5.into(), 0.7.into()]));

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