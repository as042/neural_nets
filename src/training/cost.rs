use crate::autodiff::real::operations::OperateWithReal;
use crate::autodiff::real::real_math::RealMath;
use crate::autodiff::real::Real;
use crate::network::run_results::RunResults;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostFn<T: Real, U: RealMath + OperateWithReal<T>> {
    #[default]
    MSE,
    RMSE,
    MAE,
    Custom(fn(&Vec<U>, &Vec<T>) -> U),
}

impl<T: Real, U: RealMath + OperateWithReal<T>> CostFn<T, U> {
    #[inline]
    pub fn compute(&self, output: &Vec<U>, desired_output: &Vec<T>) -> U {
        if output.len() != desired_output.len() { panic!("Output and desired output must have same len") };
        if output.is_empty() { panic!("Len cannot be 0") };
        match self {
            CostFn::MSE => CostFn::mse(output, desired_output),
            CostFn::RMSE => CostFn::rmse(output, desired_output),
            CostFn::MAE => CostFn::mae(output, desired_output),
            CostFn::Custom(f) => f(output, desired_output),
        }
    }

    #[inline]
    pub fn mse(output: &Vec<U>, desired_output: &Vec<T>) -> U {
        let mut sum = (output[0] - desired_output[0]).powf(T::one() + T::one());
        let mut len = T::one();
        for i in 1..output.len() {
            sum = sum + (output[i] - desired_output[i]).powf(T::one() + T::one());
            len = len + T::one();
        }

        sum / len
    }

    #[inline]
    pub fn rmse(output: &Vec<U>, desired_output: &Vec<T>) -> U {
        let mut sum = (output[0] - desired_output[0]).powf(T::one() + T::one());
        let mut len = T::one();
        for i in 1..output.len() {
            sum = sum + (output[i] - desired_output[i]).powf(T::one() + T::one());
            len = len + T::one();
        }

        (sum / len).sqrt()
    }

    #[inline]
    pub fn mae(output: &Vec<U>, desired_output: &Vec<T>) -> U {
        let mut sum = (output[0] - desired_output[0]).abs();
        let mut len = T::one();
        for i in 1..output.len() {
            sum = sum + (output[i] - desired_output[i]).abs();
            len = len + T::one();
        }

        sum / len
    }
}

impl<T: Real, U: RealMath + OperateWithReal<T>> RunResults<T, U> {
    #[inline]
    pub fn cost(&mut self, cost_fn: &CostFn<T, U>, desired_output: &Vec<T>) -> U {
        if self.output().len() != desired_output.len() {
            panic!("Desired output len must be same as output len");
        }

        cost_fn.compute(&self.output(), desired_output)
    }
}

#[test]
fn cost_test() {
    assert_eq!(CostFn::mse(&vec![0.5, 0.2, -0.3], &vec![0.4, 0.0, 0.3]), 0.41 / 3.0);
    assert_eq!(CostFn::rmse(&vec![0.5, 0.2, -0.3], &vec![0.4, 0.0, 0.3]), (0.41 / 3f64).sqrt());
    assert_eq!(CostFn::mae(&vec![0.5, 0.2, -0.3], &vec![0.4, 0.0, 0.3]), 0.3);
    assert_eq!((CostFn::MSE.compute(&vec![0.5, 0.2, -0.2], &vec![0.4, 0.0, 0.3]) * 1000f64).round() / 1000.0, 0.1);
}