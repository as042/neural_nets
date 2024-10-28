use crate::autodiff::real::Real;

use super::operations::{BinaryOperations, Clamp, UnaryOperations};

impl Real for f32 {
    const MIN: Self = f32::MIN;
    
    const MAX: Self = f32::MAX;

    #[inline]
    fn zero() -> Self {
        0f32
    }

    #[inline]
    fn one() -> Self {
        1f32
    }
}

impl UnaryOperations for f32 {
    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn signum(self) -> Self {
        self.signum()
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }

    #[inline]
    fn exp2(self) -> Self {
        self.exp2()
    }

    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        self.log2()
    }

    #[inline]
    fn log10(self) -> Self {
        self.log10()
    }

    #[inline]
    fn cbrt(self) -> Self {
        self.cbrt()
    }

    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }

    #[inline]
    fn asin(self) -> Self {
        self.asin()
    }

    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }

    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }

    #[inline]
    fn exp_m1(self) -> Self {
        self.exp_m1()
    }

    #[inline]
    fn ln_1p(self) -> Self {
        self.ln_1p()
    }

    #[inline]
    fn sinh(self) -> Self {
        self.sinh()
    }

    #[inline]
    fn cosh(self) -> Self {
        self.cosh()
    }

    #[inline]
    fn tanh(self) -> Self {
        self.tanh()
    }

    #[inline]
    fn asinh(self) -> Self {
        self.asinh()
    }

    #[inline]
    fn acosh(self) -> Self {
        self.acosh()
    }

    #[inline]
    fn atanh(self) -> Self {
        self.atanh()
    }
    
    #[inline]
    fn trunc(self) -> Self {
        self.trunc()
    }
    
    #[inline]
    fn floor(self) -> Self {
        self.floor()
    }
    
    #[inline]
    fn ceil(self) -> Self {
        self.ceil()
    }

    #[inline]
    fn round(self) -> Self {
        self.round()
    }
}

impl BinaryOperations for f32 {
    #[inline]
    fn log(self, base: Self) -> Self {
        self.log(base)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
}

impl Clamp for f32 {
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}