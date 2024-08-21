pub mod network;
pub mod training;
pub mod input_neuron;
pub mod layer;
pub mod neuron;
pub mod weight;
pub mod network_builder;

pub mod prelude {
    pub use crate::{network::*, training::*, network_builder::*, layer::*};
}