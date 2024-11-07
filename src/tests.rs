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
        .feed_forward_layer(ActivationFn::ReLU, 10)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = identity_data_set();

    let params = net.random_params::<f64>(Seed::Input(1.0));

    let res = net.run(&vec![0.1], &params);
    println!("res: {:?}", res);

    let optimized = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(10000)
        .num_epochs(2000)
        .weight_max(10.0)
        .eta(Eta::Decreasing(0.1, 0.001))
        .stoch_shuffle_seed(Seed::OS)
        .train();

    let res = net.run(&vec![0.1], &optimized);
    println!("res: {:?}", res);
    println!("new params: {:?}", optimized);

    panic!();
}