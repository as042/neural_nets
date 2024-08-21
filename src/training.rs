use core::f64;
use std::f64::consts::{E, PI};
use autodiff::*;

use crate::{layer::Layer, network::Network};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActivationFunction {
    #[default]
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    SiLU,
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunSettings {
    input: Vec<FT<f64>>,
    activation_fn: ActivationFunction,
    clamp: bool,
    print: bool,
}

impl RunSettings {
    /// Creates a new `Self` with the given input and activation function.
    #[inline]
    pub fn new(input: Vec<f64>, activation_fn: ActivationFunction, clamp: bool) -> Self {
        RunSettings {
            input: input.iter().map(|f| F::new(*f, 0.0)).collect(),
            activation_fn,
            clamp,
            print: false,
        }
    }

    /// Creates a new `Self` with the given input and activation function and has printing enabled.
    #[inline]
    pub fn new_with_print(input: Vec<f64>, activation_fn: ActivationFunction, clamp: bool) -> Self {
        RunSettings {
            input: input.iter().map(|f| F::new(*f, 0.0)).collect(),
            activation_fn,
            clamp,
            print: true,
        }
    }
}

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
    /// Runs `self` with the given input.
    #[inline]
    pub fn run(&mut self, settings: &RunSettings) {
        let input = &settings.input;

        assert_eq!(input.len(), self.input_layer.len());

        // set input layer
        for i in 0..input.len() {
            self.input_layer[i].activation = input[i];
        }

        // compute first layer
        for n in 0..self.layers[0].neurons {
            let mut sum = self.neurons[n].bias;

            for w in 0..self.input_layer.len() {
                sum += self.weights[self.neurons[n].weight_start_idx + w].value() * self.input_layer[w].activation;
            }

            self.neurons[n].activation = match settings.activation_fn {
                ActivationFunction::Sigmoid => Self::sigmoid(sum),
                ActivationFunction::Tanh => sum.tanh(),
                ActivationFunction::ReLU => Self::relu(sum),
                ActivationFunction::GELU => Self::gelu(sum),
                ActivationFunction::SiLU => Self::silu(sum),
            }
        }

        // compute all other layers
        for l in 1..self.layers.len() {
            for n in 0..self.layers[l].neurons {
                let neuron_idx = self.layer(l).neuron_start_idx + n;

                let mut sum = self.neurons[neuron_idx].bias;

                for w in 0..self.layers[l - 1].neurons {
                    sum += self.weights[self.neurons[neuron_idx].weight_start_idx + w].value() * self.neurons[self.layers[l - 1].neuron_start_idx + w].activation;
                }

                self.neurons[neuron_idx].activation = match settings.activation_fn {
                    ActivationFunction::Sigmoid => Self::sigmoid(sum),
                    ActivationFunction::Tanh => sum.tanh(),
                    ActivationFunction::ReLU => Self::relu(sum),
                    ActivationFunction::GELU => Self::gelu(sum),
                    ActivationFunction::SiLU => Self::silu(sum),
                }
            }
        }

        if settings.print {
            println!("Output: {:?}", self.output());
        }
    }

    #[inline]
    fn run_diff(x: &[FT<f64>]) -> FT<f64> {
        let mut net = Network::new();
        
        let nan_idx = x.iter().position(|x| x.is_nan()).unwrap();

        if x[nan_idx].deriv() != 0.0 {
            return F::zero();
        }

        for l in 0..nan_idx {
            net.add_layer(Layer::new().add_neurons(x[l].to_usize().unwrap()));
        }

        let mut net = net.build();

        for w in 0..net.weights().len() {
            net.weights[w].value = x[w + nan_idx + 1];
        }
        for b in 0..net.neurons().len() {
            net.neurons[b].bias = x[b + nan_idx + 1 + net.weights().len()];
        }

        let mut input = Vec::<FT<f64>>::default();
        for i in 0..net.input_layer.len() {
            input.push(x[i + nan_idx + 1 + net.weights().len() + net.neurons().len()]);
        }
        let mut desired_output = Vec::<FT<f64>>::default();
        for o in 0..net.last_layer().neurons() {
            desired_output.push(x[o + nan_idx + 1 + net.weights().len() + net.neurons().len() + net.input_layer.len()]);
        }

        let settings = RunSettings {
            input,
            activation_fn: match x.last().unwrap().to_i32().unwrap().abs() {
                1 => ActivationFunction::Sigmoid,
                2 => ActivationFunction::Tanh,
                3 => ActivationFunction::ReLU,
                4 => ActivationFunction::GELU,
                5 => ActivationFunction::SiLU,
                _ => panic!("Invalid value"),
            },
            clamp: false,
            print: match x.last().unwrap().to_i32().unwrap().signum() {
                -1 => true,
                1 => false,
                _ => panic!(),
            },
        };

        net.run(&settings);

        let total_cost = net.total_cost(&desired_output).into();

        if settings.print {
            println!("Total cost: {}", total_cost);
        }

        total_cost
    }

    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, settings: &RunSettings, desired_output: &Vec<f64>, eta: f64) -> TrainingResults {
        let input = &settings.input;
        let desired_output: Vec<F<f64, f64>> = desired_output.iter().map(|f| F::new(*f, 0.0)).collect();
        let layers_len = self.layers().len();

        // this list transfers all necessary paramters to the run function
        let mut x = Vec::<f64>::default();

        // add layer size information
        x.push(self.input_layer.len() as f64);
        for l in 0..layers_len {
            x.push(self.layer(l).neurons() as f64);
        }
        
        // a useless identifer to indicate where the layer size info ends
        x.push(f64::NAN);

        // add all actual nn params
        for w in self.weights() {
            x.push(w.value().into());
        }
        for n in self.neurons() {
            x.push(n.bias().into());
        }
        // add the training data
        for i in input {
            x.push(i.clone().into());
        }
        for o in &desired_output {
            x.push(o.clone().into());
        }

        // add necessary settings
        x.push(match settings.activation_fn {
            ActivationFunction::Sigmoid => 1.0,
            ActivationFunction::Tanh => 2.0,
            ActivationFunction::ReLU => 3.0,
            ActivationFunction::GELU => 4.0,
            ActivationFunction::SiLU => 5.0,
        } * if settings.print { -1.0 } else { 1.0 });

        self.run(settings);

        let grad = grad(Self::run_diff, x.as_slice());

        let weights_len = self.weights().len();
        for w in 0..weights_len {
            self.weights[w].value -= eta * grad[w + 2 + layers_len];
            if settings.clamp { self.weights[w].value = self.weights[w].value.clamp(-F::one(), F::one()); }
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias -= eta * grad[b + weights_len + 2 + layers_len];
            if settings.clamp { self.neurons[b].bias = self.neurons[b].bias.clamp(-F::one(), F::one()); }
        }

        TrainingResults {
            grad,
            output: self.output().iter().map(|f| f.to_f64().unwrap()).collect(),
            cost: self.total_cost(&desired_output).to_f64().unwrap(),
        }
    }

    /// Computes square error.
    #[inline]
    fn cost(output: FT<f64>, desired_output: FT<f64>) -> FT<f64> {
        (output - desired_output).powf(F::new(2.0, 0.0))
    }

    /// Computes total square error of `self`.
    #[inline]
    pub fn total_cost(&self, desired_output: &Vec<FT<f64>>) -> FT<f64> {
        let mut total_cost = F::default();
        let output = self.output();

        if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

        for j in 0..output.len() {
            total_cost += Self::cost(output[j], desired_output[j]);
        }

        total_cost
    }

    /// Computes the sigmoid "squishification" function of the given value.
    #[inline]
    pub fn sigmoid(x: FT<f64>) -> FT<f64> {
        1.0 / (1.0 + F::new(E, 0.0).powf(-x))
    }

    /// Computes the rectified linear unit "ReLU" activation function of the given value.
    #[inline]
    pub fn relu(x: FT<f64>) -> FT<f64> {
        (x + x.abs()) / 2.0
    }

    /// Computes the Gaussian-error linear unit "GELU" activation function of the given value.
    #[inline]
    pub fn gelu(x: FT<f64>) -> FT<f64> {
        x * Self::cdf_nd(x)
    }

    /// Computes the sigmoid linear unit "SiLU" or "swish" activation function of the given value.
    #[inline]
    pub fn silu(x: FT<f64>) -> FT<f64> {
        x * Self::sigmoid(x)
    }    

    /// Computes the CDF of the standard normal distribution of the given value.
    #[inline]
    pub fn cdf_nd(x: FT<f64>) -> FT<f64> {
        0.5 * (1.0 + Self::erf(x / 2.0.sqrt()))
    }

    /// Computes the error function of the given value.
    #[inline]
    pub fn erf(x: FT<f64>) -> FT<f64> {
        2.0 / PI.sqrt() *
        x.signum() * 
        (1.0f64 - F::new(E, 0.0).powf(-x.powf(2.0.into()))).sqrt() *
        // this is an approximation derived from the Bürmann series
        (
            (
                32836.0 * F::new(E, 0.0).powf(-3.0 * x.powf(2.0.into())) -
                93678.0 * F::new(E, 0.0).powf(-2.0 * x.powf(2.0.into())) -
                5509.0 * F::new(E, 0.0).powf(-4.0 * x.powf(2.0.into())) +
                272164.0 * F::new(E, 0.0).powf(-x.powf(2.0.into())) -
                205813.0
            ) / 1935360.0 + 1.0
        )
    }
}

#[test]
fn simple_network_test() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(3))
        .add_layer(Layer::new().add_neurons(3))
        .add_layer(Layer::new().add_neurons(2))
        .build();

    net.randomize_params(Some(1));

    let settings = &RunSettings::new(
        vec![0.21, 0.1, -0.39], 
        ActivationFunction::Sigmoid,
        false
    );

    for _ in 0..10000 {
        net.train(settings, 
            &vec![0.59, 0.7], 
            0.1);
    }

    assert_eq!(net.train(settings, &vec![0.59, 0.7], 0.1).cost(), 4.5606021083089745e-31);
}