use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use super::operations::{BinaryOperations, UnaryOperations};

pub trait RealMath<Output = Self>: Clone + Copy + UnaryOperations + BinaryOperations + 
    Neg<Output = Output> + Add<Output = Output> + Sub<Output = Output> + Mul<Output = Output> + Div<Output = Output> + Rem<Output = Output> {
}

impl<U: Clone + Copy + UnaryOperations + BinaryOperations + 
    Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self>> RealMath for U {
}