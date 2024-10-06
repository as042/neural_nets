use neural_nets::prelude::*;
use training_settings::TrainingSettings;

fn main() { 
    let layout = Layout::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::ReLU, 3)
        .feed_forward_layer(ActivationFn::Linear, 4)
        .build();

    let mut param_helper = ParamHelper::<f64>::new();
    let params = param_helper.random_params(&layout, Seed::OS);

    let mut net = Network::<f64>::new(&layout, params);

    net.train(&TrainingSettings {
        batch_size: 1,
        num_epochs: 1,
        cost_fn: CostFn::MSE,
        clamp_settings: ClampSettings::new(-1.0, 1.0, -1.0, 1.0),
        eta: Eta::new(0.1),
        input_set: vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]],
        output_set: vec![vec![0.1, -0.2, 0.3, -0.5]],
    }, &mut param_helper);
    println!("{}", net);
}