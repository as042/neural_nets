use std::marker::PhantomData;

use crate::autodiff::real::{real_math::RealMath, Real};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunResults<T: Real, U: RealMath> {
    pub(super) output: Vec<U>,
    pub(super) _marker: PhantomData<T>,
}

impl<T: Real, U: RealMath> RunResults<T, U> {
    #[inline]
    pub fn output(&self) -> &Vec<U> {
        &self.output
    }
}