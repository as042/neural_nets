use crate::autodiff::{grad_num::GradNum, var::Var};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Params<'t, T: GradNum> {
    pub(super) weights: Vec<Var<'t, T>>,
    pub(super) biases: Vec<Var<'t, T>>,
    pub(super) others: Vec<Var<'t, T>>, // currently unused
}

impl<'t, T: GradNum> Params<'t, T> {
    #[inline]
    pub fn weights(&self) -> &Vec<Var<'t, T>> {
        &self.weights
    }

    #[inline]
    pub fn biases(&self) -> &Vec<Var<'t, T>> {
        &self.biases
    }

    #[inline]
    pub fn others(&self) -> &Vec<Var<'t, T>> {
        &self.others
    }
}