use crate::autodiff::{real::Real, var::Var};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunResults<T: Real> {
    pub(super) output: Vec<T>,
}

impl<T: Real> RunResults<T> {
    #[inline]
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }
}