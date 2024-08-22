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
    /// # Examples
    /// ```
    /// let mut net = Network::new()
    ///     .add_layer(Layer::new().add_neurons(1))
    ///     .add_layer(Layer::new().add_neurons(3))
    ///     .add_layer(Layer::new().add_neurons(5))
    ///     .add_layer(Layer::new().add_neurons(9))
    ///     .add_layer(Layer::new().add_neurons(3))
    ///     .build();
    ///
    /// net.randomize_params(None);
    ///
    /// let settings = &RunSettings::new(
    ///     vec![0.2], 
    ///     ActivationFunction::Tanh,
    ///     true
    /// );
    /// let desired_output = vec![0.5, 0.7, 0.56];
    ///
    /// net.run(settings);
    /// let init_cost = net.total_cost(&desired_output.iter().map(|f| F::new(*f, 0.0)).collect());
    ///
    /// for _ in 0..1000 {
    ///     println!("{:?}", net.train(settings, 
    ///         &vec![0.5, 0.7, 0.56], 
    ///         0.1).cost()
    ///     );
    /// }
    ///
    /// println!("{init_cost}");
    /// ```
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
    pub fn output(&self) -> Vec<FT<f64>> {
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
            self.weights[w].value = rng.gen_range(-2.0 - 0.01 * w as f64..2.0 + 0.01 * w as f64).into();
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias = rng.gen_range(-2.0 - 0.02 * b as f64..2.0 + 0.02 * b as f64).into();
        }
    }

    /// Sets the weights and biases of a specific neuron.
    #[inline]
    pub fn set_neuron_params(&mut self, neuron_idx: usize, bias: FT<f64>, weights: Vec<FT<f64>>) {
        assert_eq!(weights.len(), self.neurons[neuron_idx].weights);

        self.neurons[neuron_idx].bias = bias;

        for w in 0..self.neurons[neuron_idx].weights {
            self.weights[self.neurons[neuron_idx].weight_start_idx + w].value = weights[w];
        }
    }
}

#[test]
fn test_set_neuron_params() {
    let mut net = Network::new()
    .add_layer(Layer::new().add_neurons(1))
    .add_layer(Layer::new().add_neurons(1))
    .build();

    net.set_neuron_params(0, FT::new(0.1, 0.0), vec![F::new(0.3, 0.0)]);

    assert_eq!(net, 
        Network { 
            input_layer: vec![InputNeuron::default()], 
            layers: vec![Layer { neurons: 1, ..Default::default() }], 
            neurons: vec![Neuron { bias: F::new(0.1, 0.0), weights: 1, ..Default::default() }], 
            weights: vec![Weight { value: F::new(0.3, 0.0) }],
        }
    );
}

#[test]
fn test_randomize_params() {
    let mut net = Network::new()
        .add_layer(Layer::new().add_neurons(1))
        .add_layer(Layer::new().add_neurons(1))
        .build();

    net.randomize_params(Some(0));

    assert_eq!(net, 
        Network { 
            input_layer: vec![InputNeuron::default()], 
            layers: vec![Layer { neurons: 1, ..Default::default() }], 
            neurons: vec![Neuron { bias: F::new(-0.1363131108415594, 0.0), weights: 1, ..Default::default() }], 
            weights: vec![Weight { value: F::new(0.8363016617062469, 0.0) }],
        }
    );
} 