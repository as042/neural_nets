pub mod network;
pub mod training;
pub mod autodiff;

pub mod prelude {
    pub use crate::{network::*, training::*};
}