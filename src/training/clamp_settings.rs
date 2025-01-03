use crate::autodiff::real::Real;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClampSettings<T: Real> {
    pub(super) weight_min: T,
    pub(super) weight_max: T,
    pub(super) bias_min: T,
    pub(super) bias_max: T,
}

impl<T: Real> ClampSettings<T> {
    pub const NO_CLAMP: Self = ClampSettings {
        weight_min: T::MIN,
        weight_max: T::MAX,
        bias_min: T::MIN,
        bias_max: T::MAX,
    };

    #[inline]
    pub fn new(weight_min: T, weight_max: T, bias_min: T, bias_max: T) -> Self {
        if weight_max <= weight_min { panic!("Weight max must be greater than weight min") };
        if bias_max <= bias_min { panic!("Bias max must be greater than bias min") };

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
        self.weight_max
    }

    #[inline]
    pub fn bias_min(&self) -> T {
        self.bias_min
    }

    #[inline]
    pub fn bias_max(&self) -> T {
        self.bias_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_new() {
        ClampSettings::new(1.0, 0.5, 0.0, 1.0);
    }
}