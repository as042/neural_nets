use crate::autodiff::grad_num::GradNum;
use crate::autodiff::var::{Powf, Var};
use crate::network::run_results::RunResults;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostFn<'t, T: GradNum> {
    #[default]
    MSE,
    RMSE,
    MAE,
    Custom(fn(&Vec<Var<'t, T>>, &Vec<T>) -> Var<'t, T>)
}

impl<'t, T: GradNum> CostFn<'t, T> {
    #[inline]
    pub fn compute(&self, output: &Vec<Var<'t, T>>, desired_output: &Vec<T>) -> Var<'t, T> {
        match self {
            CostFn::MSE => CostFn::mse(output, desired_output),
            CostFn::RMSE => CostFn::rmse(output, desired_output),
            CostFn::MAE => CostFn::mae(output, desired_output),
            CostFn::Custom(f) => f(output, desired_output),
        }
    }

    #[inline]
    pub fn mse(output: &Vec<Var<'t, T>>, desired_output: &Vec<T>) -> Var<'t, T> {
        let mut sum = (output[0] - desired_output[0]).powf(T::one() + T::one());
        for i in 1..output.len() {
            sum = sum + (output[i] - desired_output[i]).powf(T::one() + T::one());
        }

        sum
    }

    #[inline]
    pub fn rmse(output: &Vec<Var<'t, T>>, desired_output: &Vec<T>) -> Var<'t, T> {
        let mut sum = (output[0] - desired_output[0]).powf(T::one() + T::one());
        for i in 1..output.len() {
            sum = sum + (output[i] - desired_output[i]).powf(T::one() + T::one());
        }

        sum.sqrt()
    }

    #[inline]
    pub fn mae(output: &Vec<Var<'t, T>>, desired_output: &Vec<T>) -> Var<'t, T> {
        let mut sum = (output[0] - desired_output[0]).abs();
        for i in 1..output.len() {
            sum = sum + (output[i] - desired_output[i]).abs();
        }

        sum
    }
}

impl<'t, T: GradNum> RunResults<'t, T> {
    #[inline]
    pub fn cost(&mut self, cost_fn: &CostFn<'t, T>, desired_output: &Vec<T>) -> Var<'t, T> {
        if self.output().len() != desired_output.len() {
            panic!("Desired output len must be same as output len");
        }

        cost_fn.compute(&self.output_vars(), desired_output)
    }
}