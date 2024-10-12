use neural_nets::prelude::*;

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

    let params = net.random_params::<f64>(Seed::Input(1.0));

    let res = net.run(&input, &params);
    println!("res1: {:?}", res);

    let optimized = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(90)
        .eta(Eta::Const(1.0))
        .train();

    let res = net.run(&input, &optimized);
    println!("res2: {:?}", res);
    println!("new params: {:?}", optimized);
}