pub mod autodiff;
pub mod network;
pub mod rng;
pub mod save_information;
pub mod training;
pub mod tests;

pub mod prelude {
    pub use crate::network::{*, activation_fn::*, layout::*, params::*, run_results::*};
    pub use crate::rng::Seed;
    pub use crate::save_information::*;
    pub use crate::training::{*, clamp_settings::*, cost::*, data_set::*, eta::*, training_settings::*};
}