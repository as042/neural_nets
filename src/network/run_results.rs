use crate::autodiff::{grad_num::GradNum, var::Var};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunResults<'t, T: GradNum> {
    pub(super) output_vars: Vec<Var<'t, T>>,
}

impl<'t, T: GradNum> RunResults<'t, T> {
    #[inline]
    pub fn output(&self) -> Vec<T> {
        self.output_vars.iter().map(|&x| x.val()).collect()
    }

    #[inline]
    pub fn output_vars(&self) -> &Vec<Var<'t, T>> {
        &self.output_vars
    }
}