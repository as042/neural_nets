use std::ops::Mul;

use crate::autodiff::{real::{operations::OperateWithReal, real_math::RealMath, Real}, var::*};

/// Represents the function that returns the activation of a `Neuron`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActivationFn {
    #[default]
    None,
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    SiLU,
    SmoothReLU,
}

impl ActivationFn {
    /// Runs the activation function on the given sum.
    #[inline]
    pub(crate) fn compute<'t, T: Real, U: RealMath + OperateWithReal<T>>(&self, sum: U) -> U {
        match self {
            ActivationFn::None => sum * T::zero(),
            ActivationFn::Linear => sum,
            ActivationFn::Sigmoid => ActivationFn::sigmoid(sum),
            ActivationFn::Tanh => sum.tanh(),
            ActivationFn::ReLU => ActivationFn::relu(sum),
            ActivationFn::GELU => ActivationFn::gelu(sum),
            ActivationFn::SiLU => ActivationFn::silu(sum),
            ActivationFn::SmoothReLU => ActivationFn::smooth_relu(sum),
        }
    }

    /// Computes the sigmoid "squishification" function.
    #[inline]
    pub fn sigmoid<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        ((-x).exp() + T::one()).recip()
    }

    /// Computes the rectified linear unit "ReLU" activation function.
    #[inline]
    pub fn relu<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        (x + x.abs()) / (T::one() + T::one())
    }

    /// Computes the Gaussian-error linear unit "GELU" activation function.
    #[inline]
    pub fn gelu<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        x * Self::cdf_nd(x)
    }

    /// Computes the sigmoid linear unit "SiLU" or "swish" activation function.
    #[inline]
    pub fn silu<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        x * Self::sigmoid(x)
    }    

    /// Computes the CDF of the standard normal distribution.
    #[inline]
    pub fn cdf_nd<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        let two = T::one() + T::one();
        (Self::erf(x / two.sqrt()) + T::one()) / two
    }

    /// Computes the error function.
    #[inline]
    pub fn erf<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        let one = T::one();
        let two = one + one;
        let three = two + one;
        let four = three + one;
        let five = four + one;
        let six = five + one;
        let seven = six + one;
        let eight = seven + one;
        let nine = eight + one;
        let ten = nine + one;
        let sqrtpi = 
            (ten + seven) * 
            ((ten * ten + one) * seven * four + nine) * 
            ((ten * seven + three) * (ten * three + one) * (seven * four + one) * eight * seven - ((ten + one) * three)) / ten.powf(ten + one);
        x.signum() *
        two / sqrtpi * 
        (-(-x.powf(two)).exp() + one).sqrt() *
        // this is an approximation derived from the Bürmann series
        (
            (
                (x.powf(two) * -three).exp() * (two.powf(ten + five) + (ten + five + two) * four) - // 32836.0 
                (x.powf(two) * -two).exp() * ((eight * three * five * five + one) * six * (ten + three)) - // 93678
                (x.powf(two) * -four).exp() * (((ten + one) * nine * eight - five) * seven) + // 5509
                (-x.powf(two)).exp() * ((two.powf(ten + six) + (eight * seven * three - one) * five * three) * four) - // 272164
                (eight * five * five + one) * (seven * four + one) * (seven * seven - two) // 205813
            ) / (two.powf(ten + one) * nine * seven * five * three) + one // 1935360
        )
    }

    /// Computes the SmoothReLU "Softplus" activation function.
    #[inline]
    pub fn smooth_relu<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        (x.exp() + T::one()).ln()
    }    
}