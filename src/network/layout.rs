use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::activation_fn::ActivationFn;
use super::layer::{Layer, LayerType};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Layout {
    pub(super) layers: Vec<Layer>,
}

impl Layout {
    #[inline]
    pub fn new(layers: &Vec<Layer>) -> Self {
        Layout { 
            layers: layers.to_vec(),
        }
    }

    #[inline]
    pub fn builder() -> LayoutBuilder {
        LayoutBuilder::new()
    }

    #[inline]
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    #[inline]
    pub fn num_weights(&self) -> usize {
        if self.layers.len() < 2 { return 0; }

        let mut num_weights = 0;
        for l in 1..self.layers.len() {
            if self.layers[l].layer_type != LayerType::FeedForward {
                continue;
            }

            num_weights += self.layers[l].num_neurons() * self.layers[l - 1].num_neurons();
        }

        num_weights
    }

    #[inline]
    pub fn num_biases(&self) -> usize {
        if self.layers.len() < 2 { return 0; }

        let mut num_biases = 0;
        for l in 1..self.layers.len() {
            if self.layers[l].layer_type != LayerType::FeedForward {
                continue;
            }

            num_biases += self.layers[l].num_neurons();
        }

        num_biases
    }
}

impl Display for Layout {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.layers)
    }
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct LayoutBuilder {
    layers: Vec<Layer>,
}

impl LayoutBuilder {
    #[inline]
    pub fn new() -> Self {
        LayoutBuilder::default()
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
    pub fn build(self) -> Layout {
        Layout { layers: self.layers }
    }
}