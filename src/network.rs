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