#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Weight {
    pub(crate) value: f64,
}

impl Weight {
    /// Creates a new `Weight`.
    #[inline]
    pub fn new(value: f64) -> Self {
        Weight {
            value
        }
    }

    /// Returns the value of `self`.
    #[inline]
    pub fn value(&self) -> f64{
        self.value
    }
}