use crate::autodiff::grad_num::GradNum;

/// The data returned after training a `Network`.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TrainingResults<T: GradNum> {
    grad: Vec<T>,
    output: Vec<T>,
    cost: T,
}

impl<T: GradNum> TrainingResults<T> {
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