use crate::autodiff::real::Real;

/// The data returned after training a `Network`.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TrainingResults<T: Real> {
    pub(crate) grad: Vec<T>,
    pub(crate) output: Vec<T>,
    pub(crate) cost: T,
}

impl<T: Real> TrainingResults<T> {
    /// Returns the grad of the training data.
    #[inline]
    pub fn grad(&self) -> &Vec<T> {
        &self.grad
    }

    /// Returns the output of the training data.
    #[inline]
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }

    /// Returns the cost of the training data.
    #[inline]
    pub fn cost(&self) -> T {
        self.cost
    }
}