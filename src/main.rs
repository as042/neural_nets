use neural_nets::prelude::*;

fn main() { 
    let layout = Layout::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::ReLU, 3)
        .feed_forward_layer(ActivationFn::Linear, 4)
        .build();

    let mut param_helper = ParamHelper::<f64>::new();
    let params = param_helper.default_params(&layout);

    let net = Network::new(&layout, params);

    let res = net.run(vec![0.0; 5]);

    println!("Results: {:?}", res.output());
}