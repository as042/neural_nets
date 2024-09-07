use crate::prelude::GradNum;

/// Indicates how sensitive a `Neuron` is to the activation of another `Neuron`.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Weight<T: GradNum> {
    pub(crate) value: T,
}

impl<T: GradNum> Weight<T> {
    /// Creates a new `Weight`.
    #[inline]
    pub fn new(value: T) -> Self {
        Weight {
            value
        }
    }

    /// Returns the value of `self`.
    #[inline]
    pub fn value(&self) -> T{
        self.value
    }
}