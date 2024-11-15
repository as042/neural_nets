use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use super::activation_fn::ActivationFn;

// A group of `Neuron`s with similar function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode)]
pub struct Layer {
    pub(crate) layer_type: LayerType,
    pub(crate) activation_fn: ActivationFn,
    pub(crate) num_neurons: usize,
}

impl Layer {
    #[inline]
    pub fn input(num_neurons: usize) -> Self {
        Layer { 
            layer_type: LayerType::Input, 
            activation_fn: ActivationFn::None, 
            num_neurons, 
        }
    }

    #[inline]
    pub fn feed_forward(num_neurons: usize, activation_fn: ActivationFn) -> Self {
        Layer { 
            layer_type: LayerType::FeedForward, 
            activation_fn, 
            num_neurons,
        }
    }

    /// Returns the layer type of `self`.
    #[inline]
    pub fn layer_type(self) -> LayerType {
        self.layer_type
    }

    #[inline]
    pub fn activation_fn(self) -> ActivationFn {
        self.activation_fn
    }

    /// Returns the number of neurons of `self`.
    #[inline]
    pub fn num_neurons(self) -> usize {
        self.num_neurons
    }
}

/// The type of a `Layer`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode)]
pub enum LayerType {
    #[default]
    Input,
    FeedForward,
}