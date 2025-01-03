use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::autodiff::real::{operations::OperateWithReal, real_math::RealMath, Real};

/// Represents the function that returns the activation of a `Neuron`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode)]
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
                (x.powf(two) * -two).exp() * ((eight * three * five * five + one) * six * two * (ten + three) - (ten * seven + eight)) - // 93678
                (x.powf(two) * -four).exp() * (((ten + one) * nine * eight - five) * seven) + // 5509
                (-x.powf(two)).exp() * ((two.powf(ten + six) + (eight * seven * three - one) * five * three) * four) - // 272164
                ((eight * five * five + one) * (seven * four + one) * (seven * five) + (ten * ten * nine * two - two)) // 205813
            ) / (two.powf(ten + one) * nine * seven * five * three) + one // 1935360
        )
    }

    /// Computes the SmoothReLU "Softplus" activation function.
    #[inline]
    pub fn smooth_relu<'t, T: Real, U: RealMath + OperateWithReal<T>>(x: U) -> U {
        (x.exp()).ln_1p()
    }    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(ActivationFn::sigmoid(0.0), 0.5);
        assert_eq!((ActivationFn::sigmoid(10f64) * 1E6).round() / 1E6, 0.999955);
        assert!(ActivationFn::sigmoid(f64::MIN) >= 0.0);
    }
    
    #[test]
    fn test_relu() {
        assert_eq!(ActivationFn::relu(5.0), 5.0);
        assert_eq!(ActivationFn::relu(f64::MAX / 2.0), f64::MAX / 2.0);
        assert_eq!(ActivationFn::relu(f64::EPSILON), f64::EPSILON);
        assert_eq!(ActivationFn::relu(-0.1), 0.0);
        assert_eq!(ActivationFn::relu(-1000000.0), 0.0);
    }
    
    #[test]
    fn test_gelu() {
        assert_eq!(ActivationFn::gelu(-7.32), 0.030683066768026346);
        assert_eq!(ActivationFn::gelu(-2.9), 0.004596567862235712);
        assert_eq!(ActivationFn::gelu(0.0), 0.0);
        assert_eq!(ActivationFn::gelu(2.12), 2.087044277682028);
        assert_eq!(ActivationFn::gelu(3.0), 3.0067693992198983);
    }
    
    #[test]
    fn test_cdf_nd() {
        assert_eq!(ActivationFn::cdf_nd(-1.0), 0.15864757713595257);
        assert_eq!(ActivationFn::cdf_nd(0.0), 0.5);
        assert_eq!(ActivationFn::cdf_nd(1.2), 0.8849676941167681);
    }
    
    #[test]
    fn test_erf() {
        assert_eq!(ActivationFn::erf(-1.23), -0.919123734092335);
        assert_eq!(ActivationFn::erf(0.0), 0.0);
        assert!(ActivationFn::erf(f64::MAX) < 1.01);
        assert_eq!(ActivationFn::erf(0.29), 0.31828349781690646);
    }
    
    #[test]
    fn test_smooth_relu() {
        assert!(ActivationFn::smooth_relu(f64::MIN) >= 0.0);
        assert!(ActivationFn::smooth_relu(f64::MAX).is_infinite());
        assert_eq!(ActivationFn::smooth_relu(0.0), 2f64.ln());
    }
}