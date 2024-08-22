#![allow(unused_imports)]
use std::f64::consts::E;

use neural_nets::prelude::*;

fn main() { let mut net = Network::new()
        .add_layer(Layer::new_input().add_neurons(1))
        .add_layer(Layer::new_comput().add_neurons(1000).add_activation_fn(ActivationFn::Sigmoid))
        .add_layer(Layer::new_comput().add_neurons(3).add_activation_fn(ActivationFn::Sigmoid))
        .build();

    net.randomize_params(None);

    let settings = &RunSettings::new(
        vec![0.2], 
        true
    );
    let desired_output = vec![0.5, 0.7, 0.56];

    net.run(settings);
    let init_cost = net.total_cost(&desired_output);

    for _ in 0..1000 {
        println!("{:?}", net.train(settings, 
            &vec![0.5, 0.7, 0.56], 
            0.1).cost()
        );
    }

    println!("{init_cost}");
}

// use reverse::*;

// fn foo() {
//     let tape = Tape::new();
//     let p = [2., 2.];
//     let data = [0.1, 0.9];

//     let params = tape.add_vars(&p);
//     let result = tnn(&params, &data);
//     let gradients = result.grad();

//     println!("{:?}, {:?}, {:?}", gradients.wrt(&params), tnn(&params, &data).val, ltnn(&p, &data));
// }

// fn diff_fn<'a>(params: &[Var<'a>], data: &[f64]) -> Var<'a> {
//     params[0].powf(params[1]) + data[0].sin() - params[2].asinh() / data[1]
// }

// fn ltnn<'a>(params: &[f64; 2], data: &[f64]) -> f64 {
//     let tape = Tape::new();
//     let params = tape.add_vars(params);

//     tnn(&params, data).val
// }

// fn tnn<'a>(params: &[Var<'a>], data: &[f64]) -> Var<'a> {
//     let w = params[0];
//     let b = params[1];
//     let i = data[0];
//     let o = data[1];
//     let z = w * i + b;
//     let a = 1.0 / (1.0 + (-z).exp());
//     let diff = a - o;
//     let cost = diff.powf(2.0);

//     // println!("{:?}", cost);
//     cost
// }