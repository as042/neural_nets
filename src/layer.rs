#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Layer {
    pub(crate) neurons: usize,
    pub(crate) neuron_start_idx: usize,
}

impl Layer {
    /// Creates a new `InputNeuron`.
    #[inline]
    pub fn new() -> Self {
        Layer { 
            neurons: 1, 
            ..Default::default()
        }
    }

    /// Returns the number of neurons of `self`.
    #[inline]
    pub fn neurons(&self) -> usize {
        self.neurons
    }

    /// Returns the neuron start index of `self`.
    #[inline]
    pub fn neuron_start_idx(&self) -> usize {
        self.neuron_start_idx
    }
}