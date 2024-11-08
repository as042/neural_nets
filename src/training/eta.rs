use crate::{autodiff::real::Real, prelude::ActivationFn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Eta<T: Real> {
    Const(T),
    Decreasing(T, T),
}

impl<T: Real> Eta<T> {
    /// Returns `Eta::Const(0.1)`.
    #[inline]
    pub fn point_one() -> Self {
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        Eta::Const(T::one() / ten)
    }

    /// Returns `Eta::Const(0.1)`.
    #[inline]
    pub fn point_zero_one() -> Self {
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        Eta::Const(T::one() / ten / ten)
    }

    /// Unwraps `self`, returning a tuple of both potential variants.
    #[inline]
    pub fn unwrap(&self) -> (Option<T>, Option<(T, T)>) {
        let mut inside = (None, None);
        if let Eta::Const(v) = self {
            inside.0 = Some(*v);
        }
        if let Eta::Decreasing(init, amount) = self {
            inside.1 = Some((*init, *amount));
        }

        inside
    }

    /// Returns the appropriate eta value of `self`.
    #[inline]
    pub fn val(&self, epoch: usize) -> T {
        let inside = self.unwrap();
        let mut val = T::zero();
        if let Some(v) = inside.0 {
            val = v;
        }
        if let Some((init, amount)) = inside.1 {
            val = init;
            for _ in 0..epoch {
                val = ActivationFn::eta_relu::<T, T>(val - amount);
            }
        }
        
        val
    }
}

impl<T: Real> Default for Eta<T> {
    fn default() -> Self {
        Eta::point_one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_test() {
        assert_eq!(Eta::point_one(), Eta::Const(0.1));
        assert_eq!(Eta::point_zero_one(), Eta::Const(0.01));
    
        assert_eq!(Eta::Const(0.03).unwrap(), (Some(0.03), None));
        assert_eq!(Eta::Decreasing(0.1, 0.01).unwrap(), (None, Some((0.1, 0.01))));
    
        assert_eq!(Eta::Const(0.314159).val(1), 0.314159);
        assert_eq!((Eta::Decreasing(1.0f64, 0.8).val(1) * 10.0).round() / 10.0, 0.2);
        assert!(Eta::Decreasing(1.0f64, 0.8).val(20) > 0.0 && Eta::Decreasing(1.0f64, 0.8).val(20) < 0.001);
    }
}