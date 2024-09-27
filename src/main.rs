use neural_nets::prelude::*;

fn main() { 
    // let mut net = Network::new()
    //     .add_layer(Layer::new_input().add_neurons(3))
    //     .add_layer(Layer::new_comput().add_neurons(1000000).add_activation_fn(ActivationFn::GELU))
    //     .add_layer(Layer::new_comput().add_neurons(3).add_activation_fn(ActivationFn::GELU))
    //     .build();

    // net.randomize_params(None);

    // let settings = &RunSettings::new(vec![0.2, -0.3, 0.12], true);
    // let desired_output = vec![0.5, 0.7, 0.56];

    // net.run(settings);
    // let init_cost = net.total_cost(&desired_output);

    // for _ in 0..1000 {
    //     println!("cost: {:?}", net.train(settings, &vec![0.5, 0.7, 0.56], 0.1).cost());
    // }

    // println!("initial cost: {init_cost}");
}