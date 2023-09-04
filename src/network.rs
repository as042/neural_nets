use std::f64::consts::E;
use rand::Rng;
use genetic_optimization::prelude::*;

use crate::{layer::Layer, neuron::Neuron, weight::Weight, input_neuron::InputNeuron, network_builder::NetworkBuilder};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
/// A network object. `layers` refers to non-input layers.
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

    /// Returns the output of `self`.
    #[inline]
    pub fn output(&self) -> Vec<f64> {
        let mut vec = Vec::default();

        for n in 0..self.layers.last().unwrap().neurons {
            vec.push(self.neurons[self.layers.last().unwrap().neuron_start_idx + n].activation)
        }

        vec
    }

    /// Randomizes all weights and biases of `self`.
    #[inline]
    pub fn randomize_params(&mut self) {
        let mut rng = rand::thread_rng();

        for w in 0..self.weights.len() {
            self.weights[w].value = rng.gen_range(-2.0..2.0);
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias = rng.gen_range(-2.0..2.0);
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
    pub fn run(&mut self, input: Vec<f64>) {
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

            self.neurons[n].activation = sigmoid(sum);
        }

        // compute all other layers
        for l in 1..self.layers.len() {
            for n in 0..self.layers[l].neurons {
                let mut sum = self.neurons[n].bias;
                
                for w in 0..self.layers[l - 1].neurons {
                    sum += self.weights[self.neurons[n].weight_start_idx + w].value() * self.neurons[self.layers[l - 1].neuron_start_idx + w].activation;
                }
    
                self.neurons[self.layers[l].neuron_start_idx + n].activation = sigmoid(sum);
            }
        }
    }

    /// Trains `self` using a genetic algorithm.
    #[inline]
    pub fn genetic_train(&mut self, eval: fn(Vec<f64>) -> f64) {
        let mut genome = Genome::new();
        for l in 0..self.layers.len() {
            for n in 0..self.layers[l].neurons {
                let mut chromo = Chromosome::new();
                chromo.add_gene("bias", Gene::new(self.neurons[n].bias));

                for w in 0..self.neurons[n].weights {
                    chromo.add_gene(
                        format!("l{l}n{n}(n{})w{w}(w{})", self.layers[l].neuron_start_idx + n, self.neurons[n].weight_start_idx + w), 
                        Gene::new(self.weights[self.neurons[n].weight_start_idx + w].value),
                    );
                }

                genome.add_chromosome(format!("l{l}n{n}(n{})", self.layers[l].neuron_start_idx + n), &chromo);
            }
        }
        
        let genome = genome.build();

        let optimized = Simulation::new()
            .genome(&genome)
            .eval_with_util(genetic_evaluator, eval)
            .parallelism(Parallelism::Auto)
            .print_settings(PrintSettings::PrintFull)
            .run(10);
    }
}

// sigmoid function
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[inline]
fn genetic_evaluator(genome: &Genome, nn_eval: fn(Vec<f64>) -> f64) -> f64 {
    let mut network = genome_to_network(genome);
    
    network.run(input)

    0.0
}

#[inline]
fn genome_to_network(genome: &Genome) -> Network {
    // count layers
    let mut layers = 1;
    for chromo in genome.chromosomes() {
        let layer_idx: usize = chromo.0[1..chromo.0.find('n').unwrap()].parse().unwrap();
        if layer_idx + 1 > layers {
            layers = layer_idx + 1;
        }
    }

    // parse important information
    let mut network_layout: Vec<Vec<(&String, Chromosome)>> = vec![vec![]; layers];
    for chromo in genome.chromosomes() {
        let layer_idx: usize = chromo.0[1..chromo.0.find('n').unwrap()].parse().unwrap();

        network_layout[layer_idx].push((chromo.0, chromo.1.clone()));
    }

    // build network
    let network = Network::new();
    for l in 0..network_layout.len() {
        let layer = Layer::new();

        layer.add_neurons(network_layout[l].len());

        network.add_layer(layer);
    }

    let mut network = network.build();

    // set params
    let mut neurons = 0;
    for l in 0..network_layout.len() {
        for n in 0..network_layout[l].len() {
            let mut bias = 0.0;
            let mut weights = Vec::default();

            for p in network_layout[l][n].1.genes() {
                if p.0 == "bias" {
                    bias = p.1.value();
                }
                else {
                    weights.insert(p.0[p.0.find('w').unwrap() + 1..p.0.find('(').unwrap()].parse().unwrap(), p.1.value());
                }
            }

            network.set_neuron_params(neurons, bias, weights);

            neurons += 1;
        }
    }

    network
}