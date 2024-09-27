use crate::autodiff::{grad_num::GradNum, var::*};

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
    pub(crate) fn compute<'t, T: GradNum>(&self, sum: Var<'t, T>) -> Var<'t, T> {
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

    // /// Encodes `self` as a float.
    // #[inline]
    // pub(crate) fn encode<T: GradNum>(&self) -> T {
    //     match self {
    //         ActivationFn::None => 1.0,
    //         ActivationFn::Sigmoid => 2.0,
    //         ActivationFn::Tanh => 3.0,
    //         ActivationFn::ReLU => 4.0,
    //         ActivationFn::GELU => 5.0,
    //         ActivationFn::SiLU => 6.0,
    //         ActivationFn::SmoothReLU => 7.0,
    //     }
    // }

    // /// Decodes the float back into its respective activation fn.
    // #[inline]
    // pub(crate) fn decode<T: GradNum>(float: T) -> Self {
    //     match float as i32 {
    //         1 => ActivationFn::None,
    //         2 => ActivationFn::Sigmoid,
    //         3 => ActivationFn::Tanh,
    //         4 => ActivationFn::ReLU,
    //         5 => ActivationFn::GELU,
    //         6 => ActivationFn::SiLU,
    //         7 => ActivationFn::SmoothReLU,
    //         _ => panic!("Invalid value"),
    //     }
    // }

    /// Computes the sigmoid "squishification" function.
    #[inline]
    pub fn sigmoid<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
        ((-x).exp() + T::one()).recip()
    }

    /// Computes the rectified linear unit "ReLU" activation function.
    #[inline]
    pub fn relu<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
        (x + x.abs()) / (T::one() + T::one())
    }

    /// Computes the Gaussian-error linear unit "GELU" activation function.
    #[inline]
    pub fn gelu<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
        x * Self::cdf_nd(x)
    }

    /// Computes the sigmoid linear unit "SiLU" or "swish" activation function.
    #[inline]
    pub fn silu<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
        x * Self::sigmoid(x)
    }    

    /// Computes the CDF of the standard normal distribution.
    #[inline]
    pub fn cdf_nd<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
        let two = T::one() + T::one();
        (Self::erf(x / two.sqrt()) + T::one()) / two
    }

    /// Computes the error function.
    #[inline]
    pub fn erf<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
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
    pub fn smooth_relu<'t, T: GradNum>(x: Var<'t, T>) -> Var<'t, T> {
        (x.exp() + T::one()).ln()
    }    
}

// impl<'a> ActivationFn {
//     /// Runs the differential activation function on the given sum.
//     #[inline]
//     pub(crate) fn diff_compute(&self, sum: Var<'a>) -> Var<'a> {
//         match self {
//             ActivationFn::None => sum,
//             ActivationFn::Sigmoid => ActivationFn::diff_sigmoid(sum),
//             ActivationFn::Tanh => sum.tanh(),
//             ActivationFn::ReLU => ActivationFn::diff_relu(sum),
//             ActivationFn::GELU => ActivationFn::diff_gelu(sum),
//             ActivationFn::SiLU => ActivationFn::diff_silu(sum),
//             ActivationFn::SmoothReLU => ActivationFn::diff_smooth_relu(sum),
//         }
//     }

//     /// Computes the differential sigmoid "squishification" function.
//     #[inline]
//     pub(crate) fn diff_sigmoid(x: Var<'a>) -> Var<'a> {
//         1.0 / (1.0 + (-x).exp())
//     }

//     /// Computes the differential rectified linear unit "ReLU" activation function.
//     #[inline]
//     pub(crate) fn diff_relu(x: Var<'a>) -> Var<'a> {
//         (x + x.abs()) / 2.0
//     }

//     /// Computes the differential Gaussian-error linear unit "GELU" activation function.
//     #[inline]
//     pub(crate) fn diff_gelu(x: Var<'a>) -> Var<'a> {
//         x * Self::diff_cdf_nd(x)
//     }

//     /// Computes the differential sigmoid linear unit "SiLU" or "swish" activation function.
//     #[inline]
//     pub(crate) fn diff_silu(x: Var<'a>) -> Var<'a> {
//         x * Self::diff_sigmoid(x)
//     }    

//     /// Computes the differential CDF of the standard normal distribution.
//     #[inline]
//     pub(crate) fn diff_cdf_nd(x: Var<'a>) -> Var<'a> {
//         0.5 * (1.0 + Self::diff_erf(x / 2.0T.sqrt()))
//     }

//     /// Computes the differential error function.
//     #[inline]
//     pub(crate) fn diff_erf(x: Var<'a>) -> Var<'a> {
//         2.0 / PI.sqrt() *
//         x.val().signum() * 
//         (1.0 - (-x.powf(2.0)).exp()).sqrt() *
//         // this is an approximation derived from the Bürmann series
//         (
//             (
//                 32836.0 * (-3.0 * x.powf(2.0)).exp() -
//                 93678.0 * (-2.0 * x.powf(2.0)).exp() -
//                 5509.0 * (-4.0 * x.powf(2.0)).exp() +
//                 272164.0 * (-x.powf(2.0)).exp() -
//                 205813.0
//             ) / 1935360.0 + 1.0
//         )
//     }

//     /// Computes the differential SmoothReLU "Softplus" activation function.
//     #[inline]
//     pub(crate) fn diff_smooth_relu(x: Var<'a>) -> Var<'a> {
//         (1.0T + x.exp()).ln()
//     }    
// }