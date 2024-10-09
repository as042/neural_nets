pub mod f32;
pub mod f64;
pub mod operations;
pub mod real_math;

use std::fmt::Debug;

use real_math::RealMath;

pub trait Real: RealMath + Debug + PartialEq + PartialOrd
{
    fn zero() -> Self;
    fn one() -> Self;
}