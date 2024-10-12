pub mod f32;
pub mod f64;
pub mod operations;
pub mod real_math;

use std::fmt::Debug;

use operations::Clamp;
use real_math::RealMath;

pub trait Real: RealMath + Clamp + Debug + PartialEq + PartialOrd
{
    const MIN: Self;
    const MAX: Self;

    fn zero() -> Self;
    fn one() -> Self;
}