use crate::{layer::Layer, neuron::Neuron, weight::Weight, input_neuron::InputNeuron, network_builder::NetworkBuilder};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
/// A network object. `layers` refers to non-input layers.
pub struct Network {
    input: Vec<InputNeuron>,
    layers: Vec<Layer>,
    neurons: Vec<Neuron>,
    weights: Vec<Weight>,
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
        &self.input
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

    /// Returns a mutable reference to the `Layer` at the given index.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn mut_layer(&mut self, idx: usize) -> &mut Layer {
        &mut self.layers[idx]
    }

    /// Returns a mutable reference to the `Neuron` at the given index.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn mut_neuron(&mut self, idx: usize) -> &mut Neuron {
        &mut self.neurons[idx]
    }

    /// Returns a mutable reference to the `Weight` at the given index.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn mut_weight(&mut self, idx: usize) -> &mut Weight {
        &mut self.weights[idx]
    }
}