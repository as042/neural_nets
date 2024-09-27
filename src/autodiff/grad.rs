use super::grad_num::GradNum;
use super::var::Var;

#[derive(Clone, Debug)]
pub struct Grad<T: GradNum> {
    pub(super) partials: Vec<T>,
    pub(super) num_inputs: usize,
}

impl<T: GradNum> Grad<T> {
    #[inline]
    pub fn wrt(&self, var: Var<T>) -> T {
        self.partials[var.index]
    }

    #[inline]
    pub fn full(&self) -> &Vec<T> {
        &self.partials
    }

    #[inline]
    pub fn wrt_inputs(&self) -> &[T] {
        &self.partials[0..self.num_inputs]
    }
}