pub mod activation_fn;
pub mod layer;
pub mod layout;
pub mod network_data;
pub mod params;
pub mod param_helper;
pub mod running;
pub mod run_results;

use layout::*;
use params::{Params, Seed};

use crate::autodiff::real::Real;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Network {
    layout: Layout,
}

impl Network {
    #[inline]
    pub fn new(layout: Layout) -> Self {
        // if layout.num_weights() != params.weights().len() { panic!("Number of weights must match that of layout") };
        // if layout.num_biases() != params.biases().len() { panic!("Number of biases must match that of layout") };
        // others not implemented yet

        Network { 
            layout, 
        }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[inline]
    pub fn default_params<T: Real>(&self) -> Params<T> {
        Params::default_params(self.layout())
    }

    #[inline]
    pub fn random_params<T: Real>(&self, seed: Seed<T>) -> Params<T> {
        Params::random_params(self.layout(), seed)
    }
}

// impl<'t, T: GradNum + std::fmt::Debug> std::fmt::Display for Network<'t, T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{:?}, {}", self.layout, self.params)
//     }
// }