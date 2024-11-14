#[allow(unused_imports)]
use neural_nets::prelude::*;

fn main() { 
    let net = Network::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = DataSet::builder()
        .sample(vec![0.0, 1.2, 5.1, 20.5, 4.2], vec![3.9])
        .sample(vec![0.0, 1.2, 5.1, 20.5, 4.2], vec![3.9])
        .sample(vec![0.0, 1.2, 5.1, 20.5, 4.2], vec![3.9])
        .sample(vec![0.0, 1.2, 5.1, 20.5, 4.2], vec![3.9])
        .sample(vec![0.0, 1.2, 5.1, 20.5, 4.2], vec![3.9])
        .build();

    let params = net.random_params::<f64>(Seed::OS);

    let train_res = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(10)
        .cost_fn(CostFn::MSE)
        .clamp_settings(ClampSettings::NO_CLAMP)
        .eta(Eta::Const(1E-5))
        .stoch_shuffle_seed(Seed::OS)
        .train();

    train_res.save_to_file(SaveInformation::new("training_results.ron", FileNotation::RON)).unwrap();
    println!("costs: {:?}", train_res.epoch_cost(5));
    println!("new params: {:?}", train_res.params());
}