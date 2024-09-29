pub mod network;
pub mod training;
pub mod autodiff;

pub mod prelude {
    pub use crate::network::{*, activation_fn::*, layout::*, params::*};
    pub use crate::training::*;
}