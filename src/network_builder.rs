use crate::prelude::Network;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct NetworkBuilder {
    input_neurons: usize,
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

    #[inline]
    pub fn add_layer(&self, layer_builder: LayerBuilder) -> Self {
        let mut new = self.clone();
        new.layer_builders.push(layer_builder);

        new
    }

    #[inline]
    pub fn add_layers(&self, num: usize, layer_builder: LayerBuilder) -> Self {
        let mut new = self.clone();
        for _ in 0..num {
            new.layer_builders.push(layer_builder);
        }

        new
    }

    #[inline]
    pub fn build(&self) -> Network {
        println!("{:?}", self);

        Network::default()
    }
}

impl LayerBuilder {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add_neurons(&self, neurons: usize) -> Self {
        let mut new = self.clone();
        new.neurons = neurons;

        new
    }
}