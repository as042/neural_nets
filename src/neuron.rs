use autodiff::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Neuron {
    pub(crate) activation: FT<f64>,
    pub(crate) bias: FT<f64>,
    pub(crate) weights: usize,
    pub(crate) weight_start_idx: usize,
}

impl Neuron {
    /// Creates a new `InputNeuron`.
    #[inline]
    pub fn new() -> Self {
        Neuron { 
            weights: 1, 
            ..Default::default()
        }
    }

    /// Returns the activation of `self`.
    #[inline]
    pub fn activation(&self) -> FT<f64> {
        self.activation
    }

    /// Returns the bias of `self`.
    #[inline]
    pub fn bias(&self) -> FT<f64> {
        self.bias
    }

    /// Returns the number of weights of `self`.
    #[inline]
    pub fn weights(&self) -> usize {
        self.weights
    }

    /// Returns the weight start index of `self`.
    #[inline]
    pub fn weight_start_idx(&self) -> usize {
        self.weight_start_idx
    }
}