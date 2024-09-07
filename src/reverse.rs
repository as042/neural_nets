// /// !IMPORTANT!
// /// This is the reverse crate by al-jshen. It was modified and copied into this crate.
// /// Original: https://github.com/al-jshen/reverse/tree/master

// use std::{
//     cell::RefCell,
//     fmt::Display,
//     iter::Sum,
//     ops::{Add, Div, Mul, Neg, Sub},
// };

// #[derive(Debug, Clone, Copy)]
// pub(crate) struct Node {
//     weights: [f64; 2],
//     dependencies: [usize; 2],
// }

// #[derive(Debug, Clone, Copy)]
// /// Differentiable variable. This is the main type that users will interact with.
// pub struct Var<'a> {
//     /// Value of the variable.
//     pub val: f64,
//     /// Location that can be referred to be nodes in the tape.
//     location: usize,
//     /// Reference to a tape that this variable is associated with.
//     pub tape: &'a Tape,
// }

// #[derive(Debug, Clone)]
// /// Tape (Wengert list) that tracks differentiable variables, intermediate values, and the
// /// operations applied to each.
// pub struct Tape {
//     /// Variables and operations that are tracked.
//     nodes: RefCell<Vec<Node>>,
// }

// impl Tape {
//     /// Create a new tape.
//     pub fn new() -> Self {
//         Self {
//             nodes: RefCell::new(vec![]),
//         }
//     }
//     /// Gets the number of nodes (differentiable variables and intermediate values) in the tape.
//     pub fn len(&self) -> usize {
//         self.nodes.borrow().len()
//     }
//     /// Checks whether the tape is empty.
//     pub fn is_empty(&self) -> bool {
//         self.len() == 0
//     }
//     pub(crate) fn add_node(&self, loc1: usize, loc2: usize, grad1: f64, grad2: f64) -> usize {
//         let mut nodes = self.nodes.borrow_mut();
//         let n = nodes.len();
//         nodes.push(Node {
//             weights: [grad1, grad2],
//             dependencies: [loc1, loc2],
//         });
//         n
//     }
//     /// Add a variable with value `val` to the tape. Returns a `Var<'a>` which can be used like an `f64`.
//     pub fn add_var(&self, val: f64) -> Var {
//         let len = self.len();
//         Var {
//             val,
//             location: self.add_node(len, len, 0., 0.),
//             tape: self,
//         }
//     }
//     /// Add a slice of variables to the tape. See `add_var` for details.
//     pub fn add_vars<'a>(&'a self, vals: &[f64]) -> Vec<Var<'a>> {
//         vals.iter().map(|&x| self.add_var(x)).collect()
//     }
//     /// Zero out all the gradients in the tape.
//     pub fn zero_grad(&self) {
//         self.nodes
//             .borrow_mut()
//             .iter_mut()
//             .for_each(|n| n.weights = [0., 0.]);
//     }
//     /// Clear the tape by deleting all nodes (useful for clearing out intermediate values).
//     pub fn clear(&self) {
//         self.nodes.borrow_mut().clear();
//     }
// }

// impl Default for Tape {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl<'a> Var<'a> {
//     /// Get the value of the variable.
//     pub fn val(&self) -> f64 {
//         self.val
//     }
//     /// Calculate the gradients of this variable with respect to all other (possibly intermediate)
//     /// variables that it depends on.
//     pub fn grad(&self) -> Vec<f64> {
//         let n = self.tape.len();
//         let mut derivs = vec![0.; n];
//         derivs[self.location] = 1.;

//         for (idx, n) in self.tape.nodes.borrow().iter().enumerate().rev() {
//             derivs[n.dependencies[0]] += n.weights[0].clamp(-1.0E308, 1.0E308) * derivs[idx];
//             derivs[n.dependencies[1]] += n.weights[1].clamp(-1.0E308, 1.0E308) * derivs[idx];
//         }

//         derivs
//     }
//     pub fn recip(&self) -> Self {
//         Self {
//             val: self.val.recip(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 -1. / (self.val.powi(2)),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn sin(&self) -> Self {
//         Self {
//             val: self.val.sin(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, self.val.cos(), 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn cos(&self) -> Self {
//         Self {
//             val: self.val.cos(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, -self.val.sin(), 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn tan(&self) -> Self {
//         Self {
//             val: self.val.tan(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / self.val.cos().powi(2),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn ln(&self) -> Self {
//         Self {
//             val: self.val.ln(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, 1. / self.val, 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn log(&self, base: f64) -> Self {
//         Self {
//             val: self.val.log(base),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (self.val * base.ln()),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn log10(&self) -> Self {
//         self.log(10.)
//     }
//     pub fn log2(&self) -> Self {
//         self.log(2.)
//     }
//     pub fn ln_1p(&self) -> Self {
//         Self {
//             val: self.val.ln_1p(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, 1. / (1. + self.val), 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn asin(&self) -> Self {
//         Self {
//             val: self.val.asin(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (1. - self.val.powi(2)).sqrt(),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn acos(&self) -> Self {
//         Self {
//             val: self.val.acos(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 -1. / (1. - self.val.powi(2)).sqrt(),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn atan(&self) -> Self {
//         Self {
//             val: self.val.atan(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (1. + self.val.powi(2)),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn sinh(&self) -> Self {
//         Self {
//             val: self.val.sinh(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, self.val.cosh(), 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn cosh(&self) -> Self {
//         Self {
//             val: self.val.cosh(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, self.val.sinh(), 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn tanh(&self) -> Self {
//         Self {
//             val: self.val.tanh(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (self.val.cosh().powi(2)),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn asinh(&self) -> Self {
//         Self {
//             val: self.val.asinh(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (1. + self.val.powi(2)).sqrt(),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn acosh(&self) -> Self {
//         Self {
//             val: self.val.acosh(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (self.val.powi(2) - 1.).sqrt(),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn atanh(&self) -> Self {
//         Self {
//             val: self.val.atanh(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (1. - self.val.powi(2)),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn exp(&self) -> Self {
//         Self {
//             val: self.val.exp(),
//             location: self
//                 .tape
//                 .add_node(self.location, self.location, self.val.exp(), 0.),
//             tape: self.tape,
//         }
//     }
//     pub fn exp2(self) -> Self {
//         Self {
//             val: self.val.exp2(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 self.val.exp2() * 2_f64.ln(),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn sqrt(&self) -> Self {
//         Self {
//             val: self.val.sqrt(),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 1. / (2. * self.val.sqrt()),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn cbrt(&self) -> Self {
//         self.powf(1. / 3.)
//     }
//     pub fn abs(&self) -> Self {
//         let val = self.val.abs();
//         Self {
//             val,
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 if self.val == 0. {
//                     f64::NAN
//                 } else {
//                     self.val / val
//                 },
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
//     pub fn powi(&self, n: i32) -> Self {
//         Self {
//             val: self.val.powi(n),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 n as f64 * self.val.powi(n - 1),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Display for Var<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.val)
//     }
// }

// impl<'a> PartialEq for Var<'a> {
//     fn eq(&self, other: &Self) -> bool {
//         self.val.eq(&other.val)
//     }
// }

// impl<'a> PartialOrd for Var<'a> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.val.partial_cmp(&other.val)
//     }
// }

// impl<'a> PartialEq<f64> for Var<'a> {
//     fn eq(&self, other: &f64) -> bool {
//         self.val.eq(other)
//     }
// }

// impl<'a> PartialOrd<f64> for Var<'a> {
//     fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
//         self.val.partial_cmp(other)
//     }
// }

// impl<'a> PartialEq<Var<'a>> for f64 {
//     fn eq(&self, other: &Var<'a>) -> bool {
//         other.val.eq(self)
//     }
// }

// impl<'a> PartialOrd<Var<'a>> for f64 {
//     fn partial_cmp(&self, other: &Var<'a>) -> Option<std::cmp::Ordering> {
//         other.val.partial_cmp(self)
//     }
// }

// /// Calculate gradients with respect to particular variables.
// pub trait Gradient<T, S> {
//     /// Calculate the gradient with respect to variable(s) `v`.
//     fn wrt(&self, v: T) -> S;
// }

// /// Calculate the gradient with respect to variable `v`.
// impl<'a> Gradient<&Var<'a>, f64> for Vec<f64> {
//     fn wrt(&self, v: &Var) -> f64 {
//         self[v.location]
//     }
// }

// /// Calculate the gradient with respect to all variables in `v`. Returns a vector, where the items
// /// in the vector are the gradients with respect to the variable in the original list `v`, in the
// /// same order.
// impl<'a> Gradient<&Vec<Var<'a>>, Vec<f64>> for Vec<f64> {
//     fn wrt(&self, v: &Vec<Var<'a>>) -> Vec<f64> {
//         let mut jac = vec![];
//         for i in v {
//             jac.push(self.wrt(i));
//         }
//         jac
//     }
// }

// /// Calculate the gradient with respect to all variables in `v`. Returns a vector, where the items
// /// in the vector are the gradients with respect to the variable in the original list `v`, in the
// /// same order.
// impl<'a> Gradient<&[Var<'a>], Vec<f64>> for Vec<f64> {
//     fn wrt(&self, v: &[Var<'a>]) -> Vec<f64> {
//         let mut jac = vec![];
//         for i in v {
//             jac.push(self.wrt(i));
//         }
//         jac
//     }
// }

// /// Calculate the gradient with respect to all variables in `v`. Returns a vector, where the items
// /// in the vector are the gradients with respect to the variable in the original list `v`, in the
// /// same order.
// impl<'a, const N: usize> Gradient<[Var<'a>; N], Vec<f64>> for Vec<f64> {
//     fn wrt(&self, v: [Var<'a>; N]) -> Vec<f64> {
//         let mut jac = vec![];
//         for i in v {
//             jac.push(self.wrt(&i));
//         }
//         jac
//     }
// }

// /// Calculate the gradient with respect to all variables in `v`. Returns a vector, where the items
// /// in the vector are the gradients with respect to the variable in the original list `v`, in the
// /// same order.
// impl<'a, const N: usize> Gradient<&[Var<'a>; N], Vec<f64>> for Vec<f64> {
//     fn wrt(&self, v: &[Var<'a>; N]) -> Vec<f64> {
//         let mut jac = vec![];
//         for i in v {
//             jac.push(self.wrt(i));
//         }
//         jac
//     }
// }

// impl<'a> Neg for Var<'a> {
//     type Output = Self;
//     fn neg(self) -> Self::Output {
//         self * -1.
//     }
// }

// impl<'a> Add<Var<'a>> for Var<'a> {
//     type Output = Self;
//     fn add(self, rhs: Var<'a>) -> Self::Output {
//         assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
//         Self::Output {
//             val: self.val + rhs.val,
//             location: self.tape.add_node(self.location, rhs.location, 1., 1.),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Add<f64> for Var<'a> {
//     type Output = Self;
//     fn add(self, rhs: f64) -> Self::Output {
//         Self::Output {
//             val: self.val + rhs,
//             location: self.tape.add_node(self.location, self.location, 1., 0.),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Add<Var<'a>> for f64 {
//     type Output = Var<'a>;
//     fn add(self, rhs: Var<'a>) -> Self::Output {
//         rhs + self
//     }
// }

// impl<'a> Sub<Var<'a>> for Var<'a> {
//     type Output = Self;
//     fn sub(self, rhs: Var<'a>) -> Self::Output {
//         self.add(rhs.neg())
//     }
// }

// impl<'a> Sub<f64> for Var<'a> {
//     type Output = Self;
//     fn sub(self, rhs: f64) -> Self::Output {
//         self.add(rhs.neg())
//     }
// }

// impl<'a> Sub<Var<'a>> for f64 {
//     type Output = Var<'a>;
//     fn sub(self, rhs: Var<'a>) -> Self::Output {
//         Self::Output {
//             val: self - rhs.val,
//             location: rhs.tape.add_node(rhs.location, rhs.location, 0., -1.),
//             tape: rhs.tape,
//         }
//     }
// }

// impl<'a> Mul<Var<'a>> for Var<'a> {
//     type Output = Self;
//     fn mul(self, rhs: Var<'a>) -> Self::Output {
//         assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
//         Self::Output {
//             val: self.val * rhs.val,
//             location: self
//                 .tape
//                 .add_node(self.location, rhs.location, rhs.val, self.val),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Mul<f64> for Var<'a> {
//     type Output = Self;
//     fn mul(self, rhs: f64) -> Self::Output {
//         Self::Output {
//             val: self.val * rhs,
//             location: self.tape.add_node(self.location, self.location, rhs, 0.),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Mul<Var<'a>> for f64 {
//     type Output = Var<'a>;
//     fn mul(self, rhs: Var<'a>) -> Self::Output {
//         rhs * self
//     }
// }

// impl<'a> Div<Var<'a>> for Var<'a> {
//     type Output = Self;
//     fn div(self, rhs: Var<'a>) -> Self::Output {
//         self * rhs.recip()
//     }
// }

// impl<'a> Div<f64> for Var<'a> {
//     type Output = Self;
//     fn div(self, rhs: f64) -> Self::Output {
//         self * rhs.recip()
//     }
// }

// impl<'a> Div<Var<'a>> for f64 {
//     type Output = Var<'a>;
//     fn div(self, rhs: Var<'a>) -> Self::Output {
//         Self::Output {
//             val: self / rhs.val,
//             location: rhs
//                 .tape
//                 .add_node(rhs.location, rhs.location, 0., -1. / rhs.val),
//             tape: rhs.tape,
//         }
//     }
// }

// /// Trait for calculating expressions and tracking gradients for float power operations.
// pub trait Powf<T> {
//     type Output;
//     /// Calculate `powf` for self, where `other` is the power to raise `self` to.
//     fn powf(&self, other: T) -> Self::Output;
// }

// impl<'a> Powf<Var<'a>> for Var<'a> {
//     type Output = Var<'a>;
//     fn powf(&self, rhs: Var<'a>) -> Self::Output {
//         assert_eq!(self.tape as *const Tape, rhs.tape as *const Tape);
//         Self {
//             val: self.val.powf(rhs.val),
//             location: self.tape.add_node(
//                 self.location,
//                 rhs.location,
//                 rhs.val * f64::powf(self.val, rhs.val - 1.),
//                 f64::powf(self.val, rhs.val) * f64::ln(self.val),
//             ),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Powf<f64> for Var<'a> {
//     type Output = Var<'a>;
//     fn powf(&self, n: f64) -> Self::Output {
//         Self {
//             val: f64::powf(self.val, n),
//             location: self.tape.add_node(
//                 self.location,
//                 self.location,
//                 n * f64::powf(self.val, n - 1.),
//                 0.,
//             ),
//             tape: self.tape,
//         }
//     }
// }

// impl<'a> Powf<Var<'a>> for f64 {
//     type Output = Var<'a>;
//     fn powf(&self, rhs: Var<'a>) -> Self::Output {
//         Self::Output {
//             val: f64::powf(*self, rhs.val),
//             location: rhs.tape.add_node(
//                 rhs.location,
//                 rhs.location,
//                 0.,
//                 rhs.val * f64::powf(*self, rhs.val - 1.),
//             ),
//             tape: rhs.tape,
//         }
//     }
// }

// impl<'a> Sum<Var<'a>> for Var<'a> {
//     fn sum<I: Iterator<Item = Var<'a>>>(iter: I) -> Self {
//         iter.reduce(|a, b| a + b).unwrap()
//     }
// }