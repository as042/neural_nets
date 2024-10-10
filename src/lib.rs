pub mod network;
pub mod training;
pub mod autodiff;

pub mod prelude {
    pub use crate::network::{*, activation_fn::*, layout::*, params::*, run_results::*};
    pub use crate::training::{*, clamp_settings::*, cost::*, eta::*, training_settings::*};
}