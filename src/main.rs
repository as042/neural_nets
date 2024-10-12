use neural_nets::{autodiff::var::Var, prelude::*};

fn main() { 
    let net = Network::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::ReLU, 3)
        .feed_forward_layer(ActivationFn::Linear, 4)
        .build();

    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let desired_output = vec![0.1, -0.2, 0.3, -0.5];
    let data_set = DataSet::builder()
        .sample(input.clone(), desired_output.clone())
        .sample(vec![0.9, 0.12, 0.33, 0.48, 0.55], vec![-1.1, -2.2, 0.4, -0.21])
        .sample(vec![0.54, -1.2, -0.31, 0.41, 0.53], vec![1.6, -0.5, 0.12, -0.9])
        .build();

    let params = net.random_params::<f64>(Seed::OS);

    let res = net.run(&input, params.clone());
    println!("res1: {:?}", res);

    let params = net.train::<f64, Var<f64>>(&TrainingSettings {
        batch_size: 1,
        num_epochs: 10,
        cost_fn: CostFn::MSE,
        clamp_settings: ClampSettings::new(-1.0, 1.0, -1.0, 1.0),
        eta: Eta::new(0.1),
        data_set,
    }, params);

    let res = net.run(&input, params.clone());
    println!("res2: {:?}", res);
    println!("new params: {:?}", params);
}