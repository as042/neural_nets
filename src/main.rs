use neural_nets::{autodiff::var::Var, prelude::*};
use training_settings::TrainingSettings;

fn main() { 
    let layout = Layout::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::ReLU, 3)
        .feed_forward_layer(ActivationFn::Linear, 4)
        .build();

    let net = Network::new(layout);

    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let desired_output = vec![0.1, -0.2, 0.3, -0.5];
    let params = net.random_params::<f64>(Seed::OS);

    let res = net.run(&input, params.clone());
    println!("res1: {:?}", res);

    let params = net.train::<f64, Var<f64>>(&TrainingSettings {
        batch_size: 1,
        num_epochs: 100,
        cost_fn: CostFn::MSE,
        clamp_settings: ClampSettings::new(-1.0, 1.0, -1.0, 1.0),
        eta: Eta::new(0.1),
        input_set: vec![input.clone()],
        output_set: vec![desired_output],
    }, params);

    let res = net.run(&input, params);
    println!("res2: {:?}", res);
}