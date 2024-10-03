pub mod network;
pub mod training;
pub mod autodiff;

pub mod prelude {
    pub use crate::network::{*, activation_fn::*, layout::*, params::*, param_helper::*, run_results::*, run_settings::*};
    pub use crate::training::*;
}