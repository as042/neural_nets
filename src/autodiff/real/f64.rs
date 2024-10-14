use crate::autodiff::real::Real;

use super::operations::{BinaryOperations, Clamp, UnaryOperations};

impl Real for f64 {
    const MIN: Self = f64::MIN;
    
    const MAX: Self = f64::MAX;

    fn zero() -> Self {
        0f64
    }

    fn one() -> Self {
        1f64
    }
}

impl UnaryOperations for f64 {
    fn recip(self) -> Self {
        self.recip()
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn signum(self) -> Self {
        self.signum()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn exp2(self) -> Self {
        self.exp2()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn log10(self) -> Self {
        self.log10()
    }

    fn cbrt(self) -> Self {
        self.cbrt()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn exp_m1(self) -> Self {
        self.exp_m1()
    }

    fn ln_1p(self) -> Self {
        self.ln_1p()
    }

    fn sinh(self) -> Self {
        self.sinh()
    }

    fn cosh(self) -> Self {
        self.cosh()
    }

    fn tanh(self) -> Self {
        self.tanh()
    }

    fn asinh(self) -> Self {
        self.asinh()
    }

    fn acosh(self) -> Self {
        self.acosh()
    }

    fn atanh(self) -> Self {
        self.atanh()
    }

    fn trunc(self) -> Self {
        self.trunc()
    }
    
    fn floor(self) -> Self {
        self.floor()
    }
    
    fn ceil(self) -> Self {
        self.ceil()
    }

    fn round(self) -> Self {
        self.round()
    }
}

impl BinaryOperations for f64 {
    fn log(self, base: Self) -> Self {
        self.log(base)
    }

    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
}

impl Clamp for f64 {
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}