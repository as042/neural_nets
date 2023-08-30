use crate::{prelude::{Network, Layer}, neuron::Neuron, input_neuron::InputNeuron, weight::Weight};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct NetworkBuilder {
    layer_builders: Vec<LayerBuilder>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct LayerBuilder {
    neurons: usize,
}

impl NetworkBuilder {
    /// Creates a new `NetworkBuilder`.
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds a layer to the network.
    #[inline]
    pub fn add_layer(&self, layer_builder: LayerBuilder) -> Self {
        let mut new = self.clone();
        new.layer_builders.push(layer_builder);

        new
    }

    /// Adds a layer to the network multiple times.
    #[inline]
    pub fn add_layers(&self, num: usize, layer_builder: LayerBuilder) -> Self {
        let mut new = self.clone();
        for _ in 0..num {
            new.layer_builders.push(layer_builder);
        }

        new
    }

    /// Builds the final `Network`.
    #[inline]
    pub fn build(&self) -> Network {
        assert!(self.layer_builders.len() > 1);

        let mut network = Network::default();

        for _ in 0..self.layer_builders[0].neurons {
            network.input_layer.push(InputNeuron::new());
        }

        let mut neurons = 0;
        let mut weights = 0;
        for l in 1..self.layer_builders.len() {
            let neurons_in_layer = self.layer_builders[l].neurons;
            network.layers.push(Layer { neurons: neurons_in_layer, neuron_start_idx: neurons });

            let weights_per_neuron = self.layer_builders[l - 1].neurons;
            for _ in 0..neurons_in_layer {
                network.neurons.push(Neuron { weights: weights_per_neuron, weight_start_idx: weights, ..Default::default() });
                weights += weights_per_neuron;
            }

            neurons += neurons_in_layer;
        }

        network.weights = vec![Weight::default(); weights];

        network
    }
}

impl LayerBuilder {
    /// Creates a new `LayerBuilder`.
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds neurons to the layer.
    #[inline]
    pub fn add_neurons(&self, neurons: usize) -> Self {
        let mut new = self.clone();
        new.neurons = neurons;

        new
    }
}