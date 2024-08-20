use autodiff::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct InputNeuron {
    pub(crate) activation: FT<f64>,
}

impl InputNeuron {
    /// Creates a new `InputNeuron`.
    #[inline]
    pub fn new() -> Self {
        InputNeuron::default()
    }

    /// Returns the activation of `self`.
    #[inline]
    pub fn activation(&self) -> FT<f64> {
        self.activation
    }
}