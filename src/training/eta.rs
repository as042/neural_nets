use crate::autodiff::real::Real;


// this doesn't do anything yet but eventually it will support stuff like eta decreasing over time

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Eta<T: Real> {
    val: T,
}

impl<T: Real> Eta<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        Eta { 
            val,
        }
    }

    #[inline]
    pub fn val(&self) -> T {
        self.val
    }
}