use crate::autodiff::real::Real;

use super::operations::{BinaryOperations, Clamp, UnaryOperations};

impl Real for f32 {
    fn zero() -> Self {
        todo!()
    }

    fn one() -> Self {
        todo!()
    }
}

impl UnaryOperations for f32 {
    fn recip(self) -> Self {
        self.recip()
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn signum(self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        todo!()
    }

    fn exp(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn log10(self) -> Self {
        todo!()
    }

    fn cbrt(self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn tan(self) -> Self {
        todo!()
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn exp_m1(self) -> Self {
        todo!()
    }

    fn ln_1p(self) -> Self {
        todo!()
    }

    fn sinh(self) -> Self {
        todo!()
    }

    fn cosh(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }

    fn gamma(self) -> Self {
        todo!()
    }

    fn ln_gamma(self) -> Self {
        todo!()
    }
}

impl BinaryOperations for f32 {
    fn log(self, base: Self) -> Self {
        todo!()
    }

    fn powf(self, n: Self) -> Self {
        todo!()
    }
}

impl Clamp for f32 {
    fn clamp(self, min: Self, max: Self) {
        todo!()
    }
}