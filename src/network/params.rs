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

impl<'t, T: GradNum + std::fmt::Debug> std::fmt::Display for Params<'t, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Params {{ weights: [{}], biases: [{}], others: [{}] }}", 
            self.weights.iter().fold(String::new(), |acc, &num| acc + &num.to_string() + ", "), 
            self.biases.iter().fold(String::new(), |acc, &num| acc + &num.to_string() + ", "), 
            self.others.iter().fold(String::new(), |acc, &num| acc + &num.to_string() + ", ")
        )
    }
}