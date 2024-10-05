pub mod activation_fn;
pub mod layer;
pub mod layout;
pub mod network_data;
pub mod params;
pub mod param_helper;
pub mod running;
pub mod run_results;

use crate::autodiff::grad_num::GradNum;

use layout::*;
use params::Params;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Network<'t, T: GradNum> {
    layout: &'t Layout,
    params: Params<'t, T>,
}

impl<'t, T: GradNum> Network<'t, T> {
    #[inline]
    pub fn new(layout: &'t Layout, params: Params<'t, T>) -> Self {
        if layout.num_weights() != params.weights().len() { panic!("Number of weights must match that of layout") };
        if layout.num_biases() != params.biases().len() { panic!("Number of biases must match that of layout") };
        // others not implemented yet

        Network { 
            layout, 
            params,
        }
    }

    #[inline]
    pub fn layout(&self) -> &'t Layout {
        self.layout
    }

    #[inline]
    pub fn params(&self) -> &Params<'t, T> {
        &self.params
    }
}

impl<'t, T: GradNum + std::fmt::Debug> std::fmt::Display for Network<'t, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}, {}", self.layout, self.params)
    }
}