use crate::prelude::ActivationFn;

use super::{layer::Layer, Layout, Network};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct NetworkBuilder {
    layers: Vec<Layer>,
}

impl NetworkBuilder {
    #[inline]
    pub fn new() -> Self {
        NetworkBuilder::default()
    }

    #[inline]
    pub fn input_layer(mut self, num_neurons: usize) -> Self {
        self.layers.push(Layer::input(num_neurons));
        self
    }

    #[inline]
    pub fn feed_forward_layer(mut self, activation_fn: ActivationFn, num_neurons: usize) -> Self {
        self.layers.push(Layer::feed_forward(num_neurons, activation_fn));
        self
    }

    #[inline]
    pub fn build(self) -> Network {
        Network {
            layout: Layout { layers: self.layers },
        }
    }
}