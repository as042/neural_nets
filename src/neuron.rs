#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Neuron {
    pub(crate) activation: f64,
    pub(crate) bias: f64,
    pub(crate) num_weights: usize,
    pub(crate) weight_start_idx: usize,
}

impl Neuron {
    /// Creates a new `Neuron`.
    #[inline]
    pub fn new() -> Self {
        Neuron { 
            num_weights: 1, 
            ..Default::default()
        }
    }

    /// Returns the activation of `self`.
    #[inline]
    pub fn activation(&self) -> f64 {
        self.activation
    }

    /// Returns the bias of `self`.
    #[inline]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Returns the number of weights of `self`.
    #[inline]
    pub fn num_weights(&self) -> usize {
        self.num_weights
    }

    /// Returns the weight start index of `self`.
    #[inline]
    pub fn weight_start_idx(&self) -> usize {
        self.weight_start_idx
    }
}