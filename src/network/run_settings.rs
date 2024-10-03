use crate::autodiff::grad_num::GradNum;

/// Used to configure how a `Network` is run.
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunSettings<T: GradNum> {
    pub(crate) input: Vec<T>,
    pub(crate) clamp: bool,
}

impl<'t, T: GradNum> RunSettings<T> {
    /// Creates a new `Self` with the given input and activation function.
    #[inline]
    pub fn new(input: Vec<T>, clamp: bool) -> Self {
        RunSettings {
            input,
            clamp,
        }
    }
}