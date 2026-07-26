# neural_nets

Construct feedforward neural networks of all shapes, sizes, and activation functions, and train them
with gradient descent. This is a from-scratch Rust library with no machine learning dependencies:
gradients come from a hand-written reverse-mode autodiff engine (`Tape` and `Var`) that records
operations and backpropagates through them, and everything is generic over a `Real` trait so a
network can be trained in either `f32` or `f64`. Networks are assembled with a builder, layer by
layer, and can use any of the built-in activation functions (Linear, Sigmoid, Tanh, ReLU, GELU,
SiLU, SmoothReLU). Training is mini-batch stochastic gradient descent with a choice of cost function
(MSE, RMSE, MAE, or a custom `fn`), a constant or exponentially decaying learning rate, and optional
clamping of weights and biases. Weights, layouts, and full training histories can be serialized to
binary, JSON, RON, or TOML.

## Example

```rust
use neural_nets::prelude::*;

fn main() {
    let net = Network::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::ReLU, 3)
        .feed_forward_layer(ActivationFn::Linear, 4)
        .build();

    let data_set = DataSet::builder()
        .sample(vec![0.1, 0.2, 0.3, 0.4, 0.5], vec![0.1, -0.2, 0.3, -0.5])
        .sample(vec![0.9, 0.12, 0.33, 0.48, 0.55], vec![-1.1, -2.2, 0.4, -0.21])
        .sample(vec![0.54, -1.2, -0.31, 0.41, 0.53], vec![1.6, -0.5, 0.12, -0.9])
        .build();

    let params = net.random_params::<f64>(Seed::Input(1.0));

    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    println!("before: {:?}", net.run(&input, &params).output());

    let results = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(1000)
        .cost_fn(CostFn::MSE)
        .clamp_settings(ClampSettings::NO_CLAMP)
        .eta(Eta::Decay(0.1, 0.001))
        .stoch_shuffle_seed(Seed::Input(1.0))
        .train();

    println!("after:  {:?}", net.run(&input, results.params()).output());
    println!("cost per epoch: {:?}", results.epoch_cost(5));

    results
        .save_to_file(SaveInformation::new("training_results.json", FileNotation::JSON))
        .unwrap();
}
```