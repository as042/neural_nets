#![cfg(test)]
use crate::prelude::*;

fn identity_data_set() -> DataSet<f64> {
    let mut data_set = DataSet::builder();

    for i in 0..10000 {
        data_set = data_set.sample(vec![i as f64 / 100.0], vec![i as f64 / 100.0]);
    }

    data_set.build()
}

#[test]
#[ignore]
#[should_panic]
fn linear_identity_test() {
    let net = Network::builder()
        .input_layer(1)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = identity_data_set();

    let params = net.random_params::<f64>(Seed::Input(10.0));

    let train_res = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(10000)
        .num_epochs(1000)
        .cost_fn(CostFn::MSE)
        .clamp_settings(ClampSettings::NO_CLAMP)
        .eta(Eta::Const(0.000001))
        .stoch_shuffle_seed(Seed::Input(10.0))
        .train();

    println!("costs: {:?}", train_res.epoch_cost(5));
    println!("new params: {:?}", train_res.params());
}