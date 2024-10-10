use std::ops::{Add, Div, Mul, Rem, Sub};

use super::Real;

pub trait UnaryOperations {
    fn recip(self) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn cbrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn exp_m1(self) -> Self;
    fn ln_1p(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
}

pub trait BinaryOperations<Rhs = Self, Output = Self> {
    fn log(self, base: Rhs) -> Output;
    fn powf(self, n: Rhs) -> Output;
}

pub trait Clamp<MinMax = Self> {
    fn clamp(self, min: MinMax, max: MinMax) -> Self;
}

pub trait OperateWithReal<T: Real, Output = Self>: 
    Sized + BinaryOperations<T> + 
    Add<T, Output = Output> + Sub<T, Output = Output> + Mul<T, Output = Output> + Div<T, Output = Output> + Rem<T, Output = Output> {
}