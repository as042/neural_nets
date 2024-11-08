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
fn linear_identity_test() {
    let net = Network::builder()
        .input_layer(1)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = identity_data_set();

    let params = net.random_params::<f64>(Seed::Input(10.0));

    let res = net.run(&vec![0.1], &params);
    println!("res: {:?}", res);

    let train_res = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(100)
        .weight_min(-10.0)
        .weight_max(2.0)
        .bias_min(-10.0)
        .bias_max(10.0)
        .eta(Eta::Const(0.1))
        .stoch_shuffle_seed(Seed::OS)
        .train();

    let res = net.run(&vec![0.1], &train_res.params());
    println!("res: {:?}", res);
    println!("new params: {:?}", train_res.params());
    println!("costs: {:?}", train_res.epoch_cost());

    panic!();
}