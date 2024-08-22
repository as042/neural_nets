use autodiff::*;
use std::f64::consts::{E, PI};

use crate::network::Network;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActivationFunction {
    #[default]
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    SiLU,
    SmoothReLU,
}

impl Network {
    

    /// Computes the sigmoid "squishification" function of the given value.
    #[inline]
    pub fn sigmoid(x: FT<f64>) -> FT<f64> {
        1.0 / (1.0 + F::new(E, 0.0).powf(-x))
    }

    /// Computes the rectified linear unit "ReLU" activation function of the given value.
    #[inline]
    pub fn relu(x: FT<f64>) -> FT<f64> {
        (x + x.abs()) / 2.0
    }

    /// Computes the Gaussian-error linear unit "GELU" activation function of the given value.
    #[inline]
    pub fn gelu(x: FT<f64>) -> FT<f64> {
        x * Self::cdf_nd(x)
    }

    /// Computes the sigmoid linear unit "SiLU" or "swish" activation function of the given value.
    #[inline]
    pub fn silu(x: FT<f64>) -> FT<f64> {
        x * Self::sigmoid(x)
    }    

    /// Computes the CDF of the standard normal distribution of the given value.
    #[inline]
    pub fn cdf_nd(x: FT<f64>) -> FT<f64> {
        0.5 * (1.0 + Self::erf(x / 2.0.sqrt()))
    }

    /// Computes the error function of the given value.
    #[inline]
    pub fn erf(x: FT<f64>) -> FT<f64> {
        2.0 / PI.sqrt() *
        x.signum() * 
        (1.0f64 - F::new(E, 0.0).powf(-x.powf(2.0.into()))).sqrt() *
        // this is an approximation derived from the Bürmann series
        (
            (
                32836.0 * F::new(E, 0.0).powf(-3.0 * x.powf(2.0.into())) -
                93678.0 * F::new(E, 0.0).powf(-2.0 * x.powf(2.0.into())) -
                5509.0 * F::new(E, 0.0).powf(-4.0 * x.powf(2.0.into())) +
                272164.0 * F::new(E, 0.0).powf(-x.powf(2.0.into())) -
                205813.0
            ) / 1935360.0 + 1.0
        )
    }

    /// Computes the SmoothReLU "Softplus" activation function of the given value.
    #[inline]
    pub fn smooth_relu(x: FT<f64>) -> FT<f64> {
        (1.0f64 + F::new(E, 0.0).powf(x)).ln()
    }    
}