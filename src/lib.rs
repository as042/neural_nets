pub mod network;
pub mod input_neuron;
pub mod layer;
pub mod neuron;
pub mod weight;
pub mod network_builder;

pub mod prelude {
    pub use crate::{network::*, network_builder::*};
}