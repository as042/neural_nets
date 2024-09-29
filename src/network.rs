pub mod activation_fn;
pub mod layer;
pub mod network_builder;
pub mod layout;
pub mod params;
pub mod param_helper;
pub mod running;

use crate::autodiff::grad_num::GradNum;

use layout::*;
use params::Params;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Network<'t, T: GradNum> {
    layout: Layout,
    params: Params<'t, T>,
}

impl<'t, T: GradNum> Network<'t, T> {
    #[inline]
    pub fn new(layout: Layout, params: Params<'t, T>) -> Self {
        Network { 
            layout, 
            params, 
        }
    }
}