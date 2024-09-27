pub mod grad;
pub mod grad_num;
pub mod node;
pub mod tape;
pub mod var;

use var::Var;

pub type Param<'t, T> = Var<'t, T>;