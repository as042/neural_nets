pub mod network;
pub mod running;
pub mod training;
pub mod activation_functions;
pub mod layer;
pub mod neuron;
pub mod weight;
pub mod network_builder;
pub mod reverse;
pub mod autodiff;

pub mod prelude {
    pub use crate::{network::*, running::*, training::*, network_builder::*, layer::*, activation_functions::*, autodiff::*};
}