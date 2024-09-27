use crate::autodiff::{grad_num::GradNum, var::Var};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Params<'t, T: GradNum> {
    weights: Vec<Var<'t, T>>,
    biases: Vec<Var<'t, T>>,
    others: Vec<Var<'t, T>>,
}