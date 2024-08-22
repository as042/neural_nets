use std::f64::consts::PI;

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
        match float {
            1.0 => ActivationFn::None,
            2.0 => ActivationFn::Sigmoid,
            3.0 => ActivationFn::Tanh,
            4.0 => ActivationFn::ReLU,
            5.0 => ActivationFn::GELU,
            6.0 => ActivationFn::SiLU,
            7.0 => ActivationFn::SmoothReLU,
            _ => panic!("Invalid value"),
        }
    }

    /// Computes the sigmoid "squishification" function of the given value.
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Computes the rectified linear unit "ReLU" activation function of the given value.
    #[inline]
    pub fn relu(x: f64) -> f64 {
        (x + x.abs()) / 2.0
    }

    /// Computes the Gaussian-error linear unit "GELU" activation function of the given value.
    #[inline]
    pub fn gelu(x: f64) -> f64 {
        x * Self::cdf_nd(x)
    }

    /// Computes the sigmoid linear unit "SiLU" or "swish" activation function of the given value.
    #[inline]
    pub fn silu(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }    

    /// Computes the CDF of the standard normal distribution of the given value.
    #[inline]
    pub fn cdf_nd(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0f64.sqrt()))
    }

    /// Computes the error function of the given value.
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

    /// Computes the SmoothReLU "Softplus" activation function of the given value.
    #[inline]
    pub fn smooth_relu(x: f64) -> f64 {
        (1.0f64 + x.exp()).ln()
    }    
}