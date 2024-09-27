pub mod activation_fn;
pub mod layer;
pub mod network_builder;
pub mod layout;
pub mod params;
pub mod running;

use crate::autodiff::grad_num::GradNum;

use layout::Layout;
use params::Params;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Network<'t, T: GradNum> {
    layout: Layout,
    params: Params<'t, T>,
}