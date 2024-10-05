use crate::autodiff::grad_num::GradNum;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClampSettings<T: GradNum> {
    pub weight_min: T,
    pub weight_max: T,
    pub bias_min: T,
    pub bias_max: T,
}

impl<T: GradNum> ClampSettings<T> {
    #[inline]
    pub fn weight_min(&self) -> &T {
        &self.weight_min
    }

    #[inline]
    pub fn weight_max(&self) -> &T {
        &self.weight_min
    }

    #[inline]
    pub fn bias_min(&self) -> &T {
        &self.weight_min
    }

    #[inline]
    pub fn bias_max(&self) -> &T {
        &self.weight_min
    }
}