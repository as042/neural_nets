use crate::prelude::{ActivationFn, LayerBuilder};

/// The type of a `Layer`. The first `Layer` of every `Network` must be `Input`, and all other `Layer`s must be `Comput`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerType {
    #[default]
    Input,
    Comput,
}

/// A group of `Neuron`s.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Layer {
    pub(crate) num_neurons: usize,
    pub(crate) neuron_start_idx: usize,
    pub(crate) layer_type: LayerType,
    pub(crate) activation_fn: ActivationFn,
}

impl Layer {
    /// Creates a new `LayerBuilder` representing an input layer. 
    /// There can only be one input layer, and it must be the first layer.
    #[inline]
    pub fn new_input() -> LayerBuilder {
        LayerBuilder::new_input()
    }

    /// Creates a new `LayerBuilder` representing a computational layer (hidden or output).
    /// The first layer cannot be a comput layer.
    #[inline]
    pub fn new_comput() -> LayerBuilder {
        LayerBuilder::new_comput()
    }

    /// Returns the number of neurons of `self`.
    #[inline]
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Returns the neuron start index of `self`.
    #[inline]
    pub fn neuron_start_idx(&self) -> usize {
        self.neuron_start_idx
    }

    /// Returns the layer type of `self`.
    #[inline]
    pub fn layer_type(&self) -> LayerType {
        self.layer_type
    }

    /// Returns the activation fn of `self`.
    #[inline]
    pub fn activation_fn(&self) -> ActivationFn {
        self.activation_fn
    }
}