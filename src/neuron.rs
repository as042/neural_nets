/// Has an activation that feeds into other `Neuron`s in the `Network`.
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

    /// Returns the activation.
    #[inline]
    pub fn activation(&self) -> f64 {
        self.activation
    }

    /// Returns the bias.
    #[inline]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Returns the number of weights.
    #[inline]
    pub fn num_weights(&self) -> usize {
        self.num_weights
    }

    /// Returns the weight start index.
    #[inline]
    pub fn weight_start_idx(&self) -> usize {
        self.weight_start_idx
    }
}