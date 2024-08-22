use reverse::*;

use crate::{layer::Layer, network::Network, prelude::ActivationFn, running::RunSettings};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct TrainingResults {
    grad: Vec<f64>,
    output: Vec<f64>,
    cost: f64,
}

impl TrainingResults {
    /// Returns the cost of the training data.
    #[inline]
    pub fn cost(&self) -> f64 {
        self.cost
    }
}

impl Network {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, settings: &RunSettings, desired_output: &Vec<f64>, eta: f64) -> TrainingResults {
        self.run(settings);

        let param_vec = self.create_param_vec();
        let data = self.create_data_vec(settings, desired_output);

        let tape = Tape::new();
        let params = tape.add_vars(&param_vec);

        let result = Self::diff_cost_eval(&params, &data);
        let full_gradient = result.grad();
        let grad = full_gradient.wrt(&params);

        self.adjust_params(&grad, eta, settings.clamp);

        TrainingResults {
            grad,
            output: self.output(),
            cost: self.total_cost(&desired_output),
        }
    }

    /// Creates the param vector that will be passed to the autodiff.
    #[inline]
    fn create_param_vec(&self) -> Vec<f64> {
        // this list transfers all parameters to the cost evaluator
        let mut params = Vec::<f64>::default();

        // add all actual nn params
        for w in self.weights() {
            params.push(w.value());
        }
        for n in self.neurons() {
            params.push(n.bias());
        }

        params
    }

    /// Creates the param vector that will be passed to the autodiff.
    #[inline]
    fn create_data_vec(&self, settings: &RunSettings, desired_output: &Vec<f64>) -> Vec<f64> {
        let input = &settings.input;
        
        // this list transfers all necessary paramters to the run function
        let mut data = Vec::<f64>::default();

        // add layer information
        for l in 0..self.num_layers() {
            data.push(self.nth_layer(l).num_neurons() as f64);
            if l > 0 { data.push(self.nth_layer(l).activation_fn().encode() as f64 )};
        }
        
        // a useless identifer to indicate where the layer info ends
        data.push(f64::NAN);

        // add the training data
        for i in input {
            data.push(i.clone());
        }
        for o in desired_output {
            data.push(o.clone());
        }

        // a useless identifer to indicate where the training data ends
        data.push(f64::NAN);

        // add necessary settings
        data.push(if settings.print { -1.0 } else { 1.0 });

        data
    }

    #[inline]
    fn diff_cost_eval<'a>(params: &[Var<'a>], data: &[f64]) -> Var<'a> {
        let mut net = Self::gen_net(params, data);

        params[0]
    }

    // Generates the network for backprop.
    #[inline]
    fn gen_net<'a>(params: &[Var<'a>], data: &[f64]) -> (Self, RunSettings, Vec::<f64>) {
        // obtain layer info...
        let nan_idx = data.iter().position(|x| x.is_nan()).unwrap();
        let layer_info = &data[0..nan_idx];
        let data = data.to_vec();
        data.remove(nan_idx);

        // training data... 
        let nan_idx = data.iter().position(|x| x.is_nan()).unwrap();
        let training_data = &data[0..nan_idx];

        // misc
        let print = data.last().unwrap();

        // start creating network
        let mut net = Network::new();

        // construct layers
        net.add_layer(Layer::new_input().add_neurons(layer_info[0] as usize));
        for l in (1..layer_info.len()).step_by(2) {
            net.add_layer(Layer::new_comput()
                .add_neurons(layer_info[l] as usize)
                .add_activation_fn(ActivationFn::decode(layer_info[l + 1]))
            );
        }

        let mut net = net.build();

        for w in 0..net.num_weights() {
            net.weights[w].value = params[w];
        }
        for b in 0..net.neurons().len() {
            net.neurons[b].bias = x[b + nan_idx + 1 + net.weights().len()];
        }

        let mut input = Vec::<FT<f64>>::default();
        for i in 0..net.input_layer.len() {
            input.push(x[i + nan_idx + 1 + net.weights().len() + net.neurons().len()]);
        }
        let mut desired_output = Vec::<FT<f64>>::default();
        for o in 0..net.last_layer().num_neurons() {
            desired_output.push(x[o + nan_idx + 1 + net.weights().len() + net.neurons().len() + net.input_layer.len()]);
        }

        let settings = RunSettings {
            input,
            activation_fn: match x.last().unwrap().to_i32().unwrap().abs() {
                1 => ActivationFn::Sigmoid,
                2 => ActivationFn::Tanh,
                3 => ActivationFn::ReLU,
                4 => ActivationFn::GELU,
                5 => ActivationFn::SiLU,
                6 => ActivationFn::SmoothReLU,
                _ => panic!("Invalid value"),
            },
            clamp: false,
            print: match x.last().unwrap().to_i32().unwrap().signum() {
                -1 => true,
                1 => false,
                _ => panic!(),
            },
        };

        (net, settings, desired_output)
    }

    // Differentiable function for gradient descent.
    #[inline]
    fn run_diff(x: &[FT<f64>]) -> FT<f64> {
        let nan_idx = x.iter().position(|x| x.is_nan()).unwrap();

        // this function doesn't need to be run with respect to stuff like the number of neurons in each network
        if let Some(y) = x.iter().position(|x| x.deriv() == 1.0) {
            if y <= nan_idx {
                return F::zero();
            }
        }

        let (mut net , settings, desired_output) = Self::gen_net(x, nan_idx);

        net.run(&settings);

        let total_cost = net.total_cost(&desired_output).into();

        if settings.print {
            println!("Total cost: {}", total_cost);
        }

        total_cost
    }

    /// Computes total square error of `self`.
    #[inline]
    pub fn total_cost(&self, desired_output: &Vec<f64>) -> f64{
        let mut total_cost = 0.0;
        let output = self.output();

        if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

        for j in 0..output.len() {
            total_cost += Self::cost(output[j], desired_output[j]);
        }

        total_cost
    }

    /// Computes square error.
    #[inline]
    fn cost(output: f64, desired_output: f64) -> f64 {
        (output - desired_output).powf(2.0)
    }

    // Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params(&mut self, grad: &Vec<f64>, eta: f64, clamp: bool) {
        let layers_len = self.layers().len();
        let weights_len = self.weights().len();
        for w in 0..weights_len {
            self.weights[w].value -= eta * grad[w + 2 + layers_len];
            if clamp { self.weights[w].value = self.weights[w].value.clamp(-F::one(), F::one()); }
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias -= eta * grad[b + weights_len + 2 + layers_len];
            if clamp { self.neurons[b].bias = self.neurons[b].bias.clamp(-F::one(), F::one()); }
        }
    }
}

#[test]
fn simple_network_test() {
    let mut net = Network::new()
        .add_layer(Layer::new_input().add_neurons(3))
        .add_layer(Layer::new_comput().add_neurons(3).add_activation_fn(ActivationFn::Sigmoid))
        .add_layer(Layer::new_comput().add_neurons(2).add_activation_fn(ActivationFn::Sigmoid))
        .build();

    net.randomize_params(Some(1));

    let settings = &RunSettings::new(
        vec![0.21, 0.1, -0.39], 
        false
    );

    for _ in 0..10000 {
        net.train(settings, &vec![0.59, 0.7], 0.1);
    }

    assert_eq!(net.train(settings, &vec![0.59, 0.7], 0.1).cost(), 4.5606021083089745e-31);
}