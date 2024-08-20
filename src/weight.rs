use autodiff::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Weight {
    pub(crate) value: FT<f64>,
}

impl Weight {
    /// Creates a new `InputNeuron`.
    #[inline]
    pub fn new(value: FT<f64>) -> Self {
        Weight {
            value
        }
    }

    /// Returns the value of `self`.
    #[inline]
    pub fn value(&self) -> FT<f64> {
        self.value
    }
}