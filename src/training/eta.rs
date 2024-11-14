use crate::autodiff::real::Real;

use super::i64_to_real;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Eta<T: Real> {
    Const(T),
    Decay(T, T),
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
        if let Eta::Decay(init, amount) = self {
            inside.1 = Some((*init, *amount));
        }

        inside
    }

    /// Returns the appropriate eta value of `self`.
    #[inline]
    pub fn val(&self, epoch: usize, num_epochs: usize) -> T {
        if epoch >= num_epochs { panic!("Epoch must be less than number of epochs") };

        let inside = self.unwrap();
        let mut val = T::zero();
        if let Some(v) = inside.0 {
            val = v;
        }
        if let Some((init, fin)) = inside.1 {
            let num_epochs = i64_to_real::<T>(num_epochs as i64 - 1);
            val = init;
            let factor = (fin / init).powf(num_epochs.recip());
            
            for _ in 0..epoch {
                val = val * factor;
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
    use crate::tests::RoundTo;

    use super::*;

    #[test]
    fn eta_test() {
        assert_eq!(Eta::point_one(), Eta::Const(0.1));
        assert_eq!(Eta::point_zero_one(), Eta::Const(0.01));
    
        assert_eq!(Eta::Const(0.03).unwrap(), (Some(0.03), None));
        assert_eq!(Eta::Decay(0.1, 0.01).unwrap(), (None, Some((0.1, 0.01))));
    
        assert_eq!(Eta::Const(0.314159).val(0, 1).round_to(6), 0.314159);
        assert_eq!(Eta::Decay(1.0f64, 0.8123).val(2, 3).round_to(4), 0.8123);
        assert_eq!(Eta::Decay(1.0f64, 0.0001).val(3, 4).round_to(4), 0.0001);
        assert!(Eta::Decay(1.0f64, 0.0001).val(2, 4) > 0.0001 && Eta::Decay(1.0f64, 0.0001).val(2, 4) < 1.0);
    }
}