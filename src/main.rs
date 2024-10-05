use neural_nets::prelude::*;

fn main() { 
    let layout = Layout::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::ReLU, 3)
        .feed_forward_layer(ActivationFn::Linear, 4)
        .build();

    let mut param_helper = ParamHelper::<f64>::new();
    let params = param_helper.random_params(&layout, 532.0);

    let mut net = Network::<f64>::new(&layout, params);

    println!("{}", net);
}