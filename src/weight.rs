use crate::prelude::{GradNum, Var};

/// Indicates how sensitive a `Neuron` is to the activation of another `Neuron`.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Weight<'t, T: GradNum> {
    pub(crate) value: Var<'t, T>,
}

impl<'t, T: GradNum> Weight<'t, T> {
    /// Creates a new `Weight`.
    #[inline]
    pub fn new(value: Var<'t, T>) -> Self {
        Weight {
            value
        }
    }

    /// Returns the value of `self`.
    #[inline]
    pub fn value(&self) -> Var<'t, T> {
        self.value
    }
}