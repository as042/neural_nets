use std::f64::consts::E;
use autodiff::*;

use crate::{layer::Layer, network::Network};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActivationFunction {
    #[default]
    Sigmoid,
    ReLU,
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunSettings {
    input: Vec<f64>,
    activation_fn: ActivationFunction,
    print: bool,
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

impl RunSettings {
    /// Creates a new `Self` with the given input and activation function.
    #[inline]
    pub fn new(input: Vec<f64>, activation_fn: ActivationFunction) -> Self {
        RunSettings {
            input: input,
            activation_fn,
            print: false,
        }
    }

    /// Creates a new `Self` with the given input and activation function and has printing enabled.
    #[inline]
    pub fn new_with_print(input: Vec<f64>, activation_fn: ActivationFunction) -> Self {
        RunSettings {
            input: input,
            activation_fn,
            print: true,
        }
    }
}

impl Network {    
    /// Runs `self` with the given input.
    #[inline]
    pub fn run(&mut self, input: &Vec<FT<f64>>) {
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

            self.neurons[n].activation = Self::sigmoid(sum).into();
        }

        // compute all other layers
        for l in 1..self.layers.len() {
            for n in 0..self.layers[l].neurons {
                let mut sum = self.neurons[n].bias;
                
                for w in 0..self.layers[l - 1].neurons {
                    sum += self.weights[self.neurons[n].weight_start_idx + w].value() * self.neurons[self.layers[l - 1].neuron_start_idx + w].activation;
                }
    
                self.neurons[self.layers[l].neuron_start_idx + n].activation = Self::sigmoid(sum).into();
            }
        }
    }

    #[inline]
    fn run_diff(x: &[FT<f64>]) -> FT<f64> {
        let mut net = Network::new();
        let width = x[0].to_usize().unwrap();
        let height = x[1].to_usize().unwrap();

        for _ in 0..width {
            net.add_layer(Layer::new().add_neurons(height));
        }

        let mut net = net.build();

        for w in 0..(width - 1) * height.pow(2) {
            net.weights[w].value = x[w + 2].into();
        }
        for b in 0..(width -1) * height {
            net.neurons[b].bias = x[b + 2 + (width - 1) * height.pow(2)].into();
        }

        let mut input = Vec::<FT<f64>>::default();
        for i in 0..height {
            input.push(x[i + 2 + (width - 1) * height.pow(2) + (width -1) * height].into());
        }
        let mut desired_output = Vec::<FT<f64>>::default();
        for o in 0..height {
            desired_output.push(x[o + 2 + (width - 1) * height.pow(2) + (width -1) * height + height].into());
        }

        net.run(&input);

        net.total_cost(&desired_output).into()
    }

    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, input: &Vec<FT<f64>>, desired_output: &Vec<FT<f64>>, eta: FT<f64>) -> Vec<f64> {
        let mut x = Vec::<f64>::default();

        x.push((self.layers().len() as f64 + 1.0).into());
        x.push((self.input_layer.len() as f64).into());

        for w in self.weights() {
            x.push(w.value().into());
        }
        for n in self.neurons() {
            x.push(n.bias().into());
        }
        for i in input {
            x.push(i.clone().into());
        }
        for o in desired_output {
            x.push(o.clone().into());
        }

        self.run(input);

        let grad = grad(Self::run_diff, x.as_slice());

        let weights_len = self.weights().len();
        for w in 0..weights_len {
            self.weights[w].value -= eta * grad[w + 2];
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias -= eta * grad[b + weights_len + 2];
        }

        grad
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
        F::new(1.0, 0.0) / (F::new(1.0, 0.0) + F::new(E, 0.0).powf(x * F::new(-1.0, 0.0)))
    }

    /// Computes the rectified linear unit "ReLU" activation function of the given value.
    #[inline]
    pub fn relu(x: FT<f64>) -> FT<f64> {
        (x + x.abs()) / F::new(2.0, 0.0)
    }
}