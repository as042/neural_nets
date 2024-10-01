use crate::autodiff::{grad_num::GradNum, tape::Tape};

use super::{params::Params, Layout};

#[derive(Clone, Debug, Default)]
pub struct ParamHelper<T: GradNum> {
    tape: Tape<T>,
}

impl<T: GradNum> ParamHelper<T> {
    #[inline]
    pub fn new() -> Self {
        let tape = Tape::new();
        ParamHelper { 
            tape, 
        }
    }

    #[inline]
    /// Returns a `Params<'t, T>` in which every parameter is 1.
    pub fn default_params<'t>(&'t mut self, layout: &Layout) -> Params<'t, T> {
        let num_weights = layout.num_weights();
        let num_biases = layout.num_biases();

        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..layout.num_weights() {
            weights.push(self.tape.new_var(T::one()));
        }

        let mut biases = Vec::with_capacity(num_biases);
        for _ in 0..layout.num_biases() {
            biases.push(self.tape.new_var(T::one()));
        }

        Params {
            weights,
            biases,
            others: Vec::default(),
        }
    }
}