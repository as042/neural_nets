use crate::reverse::*;
use std::f64::consts::PI;

/// Represents the function that returns the activation of a `Neuron`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActivationFn {
    #[default]
    None,
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
    pub(crate) fn compute(&self, sum: f64) -> f64 {
        match self {
            ActivationFn::None => sum,
            ActivationFn::Sigmoid => ActivationFn::sigmoid(sum),
            ActivationFn::Tanh => sum.tanh(),
            ActivationFn::ReLU => ActivationFn::relu(sum),
            ActivationFn::GELU => ActivationFn::gelu(sum),
            ActivationFn::SiLU => ActivationFn::silu(sum),
            ActivationFn::SmoothReLU => ActivationFn::smooth_relu(sum),
        }
    }

    /// Encodes `self` as a float.
    #[inline]
    pub(crate) fn encode(&self) -> f64 {
        match self {
            ActivationFn::None => 1.0,
            ActivationFn::Sigmoid => 2.0,
            ActivationFn::Tanh => 3.0,
            ActivationFn::ReLU => 4.0,
            ActivationFn::GELU => 5.0,
            ActivationFn::SiLU => 6.0,
            ActivationFn::SmoothReLU => 7.0,
        }
    }

    /// Decodes the float back into its respective activation fn.
    #[inline]
    pub(crate) fn decode(float: f64) -> Self {
        match float as i32 {
            1 => ActivationFn::None,
            2 => ActivationFn::Sigmoid,
            3 => ActivationFn::Tanh,
            4 => ActivationFn::ReLU,
            5 => ActivationFn::GELU,
            6 => ActivationFn::SiLU,
            7 => ActivationFn::SmoothReLU,
            _ => panic!("Invalid value"),
        }
    }

    /// Computes the sigmoid "squishification" function.
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Computes the rectified linear unit "ReLU" activation function.
    #[inline]
    pub fn relu(x: f64) -> f64 {
        (x + x.abs()) / 2.0
    }

    /// Computes the Gaussian-error linear unit "GELU" activation function.
    #[inline]
    pub fn gelu(x: f64) -> f64 {
        x * Self::cdf_nd(x)
    }

    /// Computes the sigmoid linear unit "SiLU" or "swish" activation function.
    #[inline]
    pub fn silu(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }    

    /// Computes the CDF of the standard normal distribution.
    #[inline]
    pub fn cdf_nd(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0f64.sqrt()))
    }

    /// Computes the error function.
    #[inline]
    pub fn erf(x: f64) -> f64 {
        2.0 / PI.sqrt() *
        x.signum() * 
        (1.0 - (-x.powf(2.0)).exp()).sqrt() *
        // this is an approximation derived from the Bürmann series
        (
            (
                32836.0 * (-3.0 * x.powf(2.0)).exp() -
                93678.0 * (-2.0 * x.powf(2.0)).exp() -
                5509.0 * (-4.0 * x.powf(2.0)).exp() +
                272164.0 * (-x.powf(2.0)).exp() -
                205813.0
            ) / 1935360.0 + 1.0
        )
    }

    /// Computes the SmoothReLU "Softplus" activation function.
    #[inline]
    pub fn smooth_relu(x: f64) -> f64 {
        (1.0f64 + x.exp()).ln()
    }    
}

impl<'a> ActivationFn {
    /// Runs the differential activation function on the given sum.
    #[inline]
    pub(crate) fn diff_compute(&self, sum: Var<'a>) -> Var<'a> {
        match self {
            ActivationFn::None => sum,
            ActivationFn::Sigmoid => ActivationFn::diff_sigmoid(sum),
            ActivationFn::Tanh => sum.tanh(),
            ActivationFn::ReLU => ActivationFn::diff_relu(sum),
            ActivationFn::GELU => ActivationFn::diff_gelu(sum),
            ActivationFn::SiLU => ActivationFn::diff_silu(sum),
            ActivationFn::SmoothReLU => ActivationFn::diff_smooth_relu(sum),
        }
    }

    /// Computes the differential sigmoid "squishification" function.
    #[inline]
    pub(crate) fn diff_sigmoid(x: Var<'a>) -> Var<'a> {
        1.0 / (1.0 + (-x).exp())
    }

    /// Computes the differential rectified linear unit "ReLU" activation function.
    #[inline]
    pub(crate) fn diff_relu(x: Var<'a>) -> Var<'a> {
        (x + x.abs()) / 2.0
    }

    /// Computes the differential Gaussian-error linear unit "GELU" activation function.
    #[inline]
    pub(crate) fn diff_gelu(x: Var<'a>) -> Var<'a> {
        x * Self::diff_cdf_nd(x)
    }

    /// Computes the differential sigmoid linear unit "SiLU" or "swish" activation function.
    #[inline]
    pub(crate) fn diff_silu(x: Var<'a>) -> Var<'a> {
        x * Self::diff_sigmoid(x)
    }    

    /// Computes the differential CDF of the standard normal distribution.
    #[inline]
    pub(crate) fn diff_cdf_nd(x: Var<'a>) -> Var<'a> {
        0.5 * (1.0 + Self::diff_erf(x / 2.0f64.sqrt()))
    }

    /// Computes the differential error function.
    #[inline]
    pub(crate) fn diff_erf(x: Var<'a>) -> Var<'a> {
        2.0 / PI.sqrt() *
        x.val().signum() * 
        (1.0 - (-x.powf(2.0)).exp()).sqrt() *
        // this is an approximation derived from the Bürmann series
        (
            (
                32836.0 * (-3.0 * x.powf(2.0)).exp() -
                93678.0 * (-2.0 * x.powf(2.0)).exp() -
                5509.0 * (-4.0 * x.powf(2.0)).exp() +
                272164.0 * (-x.powf(2.0)).exp() -
                205813.0
            ) / 1935360.0 + 1.0
        )
    }

    /// Computes the differential SmoothReLU "Softplus" activation function.
    #[inline]
    pub(crate) fn diff_smooth_relu(x: Var<'a>) -> Var<'a> {
        (1.0f64 + x.exp()).ln()
    }    
}