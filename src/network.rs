use std::f64::consts::E;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use autodiff::*;

use crate::{layer::Layer, neuron::Neuron, weight::Weight, input_neuron::InputNeuron, network_builder::NetworkBuilder};

/// A network object. `layers` refers to non-input layers.
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Network {
    pub(crate) input_layer: Vec<InputNeuron>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) neurons: Vec<Neuron>,
    pub(crate) weights: Vec<Weight>,
}

impl Network {
    /// Creates a builder helper.
    #[inline]
    pub fn new() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    /// Returns the input layer of `self`.
    #[inline]
    pub fn input(&self) -> &Vec<InputNeuron> {
        &self.input_layer
    }

    /// Returns the non-input layers of `self`.
    #[inline]
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    /// Returns the neurons of `self`.
    #[inline]
    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    /// Returns the weights of `self`.
    #[inline]
    pub fn weights(&self) -> &Vec<Weight> {
        &self.weights
    }

    /// Returns the `Layer` at the given index.
    #[inline]
    pub fn layer(&self, idx: usize) -> &Layer {
        &self.layers[idx]
    }

    /// Returns the `Neuron` at the given index.
    #[inline]
    pub fn neuron(&self, idx: usize) -> &Neuron {
        &self.neurons[idx]
    }

    /// Returns the `Weight` at the given index.
    #[inline]
    pub fn weight(&self, idx: usize) -> &Weight {
        &self.weights[idx]
    }

    /// Returns the last layer of `self`.
    #[inline]
    pub fn last_layer(&self) -> &Layer {
        self.layers.last().unwrap()
    }

    /// Returns the output of `self`.
    #[inline]
    pub fn output(&self) -> Vec<f64> {
        let mut vec = Vec::default();

        for n in 0..self.last_layer().neurons {
            vec.push(self.neurons[self.last_layer().neuron_start_idx + n].activation)
        }

        vec
    }

    /// Randomizes all weights and biases of `self`.
    #[inline]
    pub fn randomize_params(&mut self, seed: Option<u64>) {
        let mut rng;
        if seed.is_some() {
            rng = ChaCha8Rng::seed_from_u64(seed.unwrap());
        }
        else {
            let mut thread_rng = thread_rng();
            rng = ChaCha8Rng::seed_from_u64(thread_rng.gen());
        }

        for w in 0..self.weights.len() {
            
            self.weights[w].value = rng.gen_range(-2.0 - 0.01 * w as f64..2.0 + 0.01 * w as f64);
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias = rng.gen_range(-2.0 - 0.02 * b as f64..2.0 + 0.02 * b as f64);
        }
    }

    /// Sets the weights and biases of a specific neuron.
    #[inline]
    pub fn set_neuron_params(&mut self, neuron_idx: usize, bias: f64, weights: Vec<f64>) {
        assert_eq!(weights.len(), self.neurons[neuron_idx].weights);

        self.neurons[neuron_idx].bias = bias;

        for w in 0..self.neurons[neuron_idx].weights {
            self.weights[self.neurons[neuron_idx].weight_start_idx + w].value = weights[w];
        }
    }

    /// Runs `self` with the given input.
    #[inline]
    pub fn run(&mut self, input: &Vec<f64>) {
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

            self.neurons[n].activation = Self::sigmoid(sum);
        }

        // compute all other layers
        for l in 1..self.layers.len() {
            for n in 0..self.layers[l].neurons {
                let mut sum = self.neurons[n].bias;
                
                for w in 0..self.layers[l - 1].neurons {
                    sum += self.weights[self.neurons[n].weight_start_idx + w].value() * self.neurons[self.layers[l - 1].neuron_start_idx + w].activation;
                }
    
                self.neurons[self.layers[l].neuron_start_idx + n].activation = Self::sigmoid(sum);
            }
        }
    }

    #[inline]
    fn run_no_self(x: &[FT<f64>]) -> FT<f64> {
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

        let mut input = Vec::<f64>::default();
        for i in 0..height {
            input.push(x[i + 2 + (width - 1) * height.pow(2) + (width -1) * height].into());
        }

        let mut desired_output = Vec::<f64>::default();
        for o in 0..height {
            desired_output.push(x[o + 2 + (width - 1) * height.pow(2) + (width -1) * height + height].into());
        }

        net.run(&input);

        net.total_cost(&desired_output).into()
    }

    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, input: &Vec<f64>, desired_output: &Vec<f64>) -> Vec<f64> {
        let mut x = Vec::default();

        x.push((self.layers().len() as f64 + 1.0).into());
        x.push((self.input_layer.len() as f64).into());

        for w in self.weights() {
            x.push(w.value());
        }

        for n in self.neurons() {
            x.push(n.bias());
        }

        for i in input {
            x.push(i.clone());
        }

        for o in desired_output {
            x.push(o.clone());
        }

        let grad = grad(Self::run_no_self, x.as_slice());

        grad
    }

    /// Computes square error.
    #[inline]
    pub fn cost(output: f64, desired_output: f64) -> f64 {
        (output - desired_output).powf(2.0)
    }

    /// Computes total square error.
    #[inline]
    pub fn total_cost(&self, desired_output: &Vec<f64>) -> f64 {
        let mut total_cost = 0.0;
        let output = self.output();

        if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

        for j in 0..output.len() {
            total_cost += Self::cost(output[j], desired_output[j]);
        }

        total_cost
    }

    /// Computes the sigmoid "squishification" function of the given value.
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }
}