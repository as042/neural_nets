use crate::autodiff::real::Real;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Eta<T: Real> {
    Const(T),
    Delta(T, T),
}

impl<T: Real> Eta<T> {
    #[inline]
    pub fn new_const(val: T) -> Self {
        Eta::Const(val)
    }

    #[inline]
    pub fn new_delta(init: T, delta: T) -> Self {
        Eta::Delta(init, delta)
    }

    #[inline]
    pub fn point_one() -> Self {
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        Eta::Const(T::one() / ten)
    }

    #[inline]
    pub fn point_zero_one() -> Self {
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        Eta::Const(T::one() / ten / ten)
    }

    #[inline]
    pub fn inside(&self) -> (Option<T>, Option<(T, T)>) {
        let mut inside = (None, None);
        if let Eta::Const(v) = self {
            inside.0 = Some(*v);
        }
        if let Eta::Delta(init, delta) = self {
            inside.1 = Some((*init, *delta));
        }

        inside
    }

    #[inline]
    pub fn val(&self, epoch: usize) -> T {
        let inside = self.inside();
        let mut val = T::zero();
        if let Some(v) = inside.0 {
            val = v;
        }
        if let Some((init, delta)) = inside.1 {
            val = init;
            for _ in 0..epoch {
                val = val - delta;
            }
        }
        
        val
    }
}

impl<T: Real> Default for Eta<T> {
    fn default() -> Self {
        todo!()
    }
}