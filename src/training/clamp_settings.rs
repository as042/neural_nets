use crate::autodiff::real::Real;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClampSettings<T: Real> {
    weight_min: T,
    weight_max: T,
    bias_min: T,
    bias_max: T,
}

impl<T: Real> ClampSettings<T> {
    #[inline]
    pub fn new(weight_min: T, weight_max: T, bias_min: T, bias_max: T) -> Self {
        if weight_max <= weight_min {
            panic!("Weight max must be greater than weight min");
        }
        if bias_max <= bias_min {
            panic!("Bias max must be greater than bias min");
        }

        ClampSettings {
            weight_min,
            weight_max,
            bias_min,
            bias_max,
        }
    }

    #[inline]
    pub fn weight_min(&self) -> T {
        self.weight_min
    }

    #[inline]
    pub fn weight_max(&self) -> T {
        self.weight_min
    }

    #[inline]
    pub fn bias_min(&self) -> T {
        self.weight_min
    }

    #[inline]
    pub fn bias_max(&self) -> T {
        self.weight_min
    }
}