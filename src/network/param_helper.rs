// use std::time::SystemTime;

// use crate::autodiff::{grad_num::GradNum, tape::Tape};

// use super::{params::Params, Layout};

// #[derive(Clone, Debug, Default)]
// pub struct TapeContainer<T: GradNum> {
//     tape: Tape<T>,
// }

// impl<T: GradNum> TapeContainer<T> {
//     #[inline]
//     pub fn new() -> Self {
//         let tape = Tape::new();
//         TapeContainer { 
//             tape, 
//         }
//     }

//     #[inline]
//     /// Returns a `Params<'t, T>` in which every parameter is 1.
//     pub fn default_params<'t>(&'t mut self, layout: &Layout) -> Params<'t, T> {
//         self.tape = Tape::new();

//         let num_weights = layout.num_weights();
//         let num_biases = layout.num_biases();

//         let mut weights = Vec::with_capacity(num_weights);
//         for _ in 0..layout.num_weights() {
//             weights.push(self.tape.new_var(T::one()));
//         }

//         let mut biases = Vec::with_capacity(num_biases);
//         for _ in 0..layout.num_biases() {
//             biases.push(self.tape.new_var(T::one()));
//         }

//         Params {
//             weights,
//             biases,
//             others: Vec::default(),
//         }
//     }

//     #[inline]
//     pub fn params<'t>(&'t mut self, weights: Vec<T>, biases: Vec<T>, _others: Vec<T>) -> Params<'t, T> {
//         self.tape = Tape::new();

//         let mut weight_vars = Vec::with_capacity(weights.len());
//         for w in 0..weights.len() {
//             weight_vars.push(self.tape.new_var(weights[w]));
//         }

//         let mut bias_vars = Vec::with_capacity(biases.len());
//         for b in 0..biases.len() {
//             bias_vars.push(self.tape.new_var(biases[b]));
//         }

//         Params { 
//             weights: weight_vars, 
//             biases: bias_vars, 
//             others: Vec::default() 
//         }
//     }

//     #[inline]
//     pub fn random_params<'t>(&'t mut self, layout: &Layout, seed: Seed<T>) -> Params<'t, T> { 
//         self.tape = Tape::new();

//         let mut init_seed = T::one();

//         if seed == Seed::OS {
//             let system_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();

//             let two = T::one() + T::one();
//             let ten = two * two * two + two;
//             let mut val = T::zero();
//             for d in system_time.to_string().chars().map(|d| d.to_digit(10).unwrap()).rev().enumerate() {
//                 let mut ten_power = T::one();
//                 for _ in 0..d.0 {
//                     ten_power = ten_power * ten;
//                 }
//                 val = val + ten_power * match d.1 {
//                     0 => T::zero(),
//                     1 => T::one(),
//                     2 => two,
//                     3 => two + T::one(),
//                     4 => two + two,
//                     5 => two + two + T::one(),
//                     6 => ten - two - two,
//                     7 => ten - two - T::one(),
//                     8 => ten - two,
//                     9 => ten - T::one(),
//                     _ => panic!("Invalid digit"),
//                 };
//             }

//             init_seed = val;
//         }
//         else if let Seed::Input(val) = seed {
//             if val <= T::zero() { panic!("Seed must be greater than 0") };

//             init_seed = val;
//         }

//         let mut rng = lehmer_rng(init_seed);

//         let two = T::one() + T::one();
//         let two_to_the_30 = two.powf(two.powf(two + two + T::one()) - two);

//         let num_weights = layout.num_weights();
//         let num_biases = layout.num_biases();

//         let mut weight_vars = Vec::with_capacity(num_weights);
//         for _ in 0..layout.num_weights() {
//             rng = lehmer_rng(rng);
//             weight_vars.push(self.tape.new_var(rng / two_to_the_30 - T::one()));
//         }

//         let mut bias_vars = Vec::with_capacity(num_biases);
//         for _ in 0..layout.num_biases() {
//             rng = lehmer_rng(rng);
//             bias_vars.push(self.tape.new_var(rng / two_to_the_30 - T::one()));
//         }

//         Params { 
//             weights: weight_vars, 
//             biases: bias_vars, 
//             others: Vec::default() 
//         }
//     }
// }

// #[inline]
// fn lehmer_rng<T: GradNum>(state: T) -> T {
//     let one = T::one();
//     let two = T::one() + T::one();
//     let three = two + one;
//     let four = three + one;
//     let sixteen = four * four;

//     // 4^7 * 3 - (4^5) + (4 * 4 * 3 * 3) - 1
//     let num48271 = sixteen * sixteen * sixteen * four * three - sixteen * sixteen * four + sixteen * three * three - one;
//     let num0x7fffffff = sixteen * sixteen * sixteen * sixteen * sixteen * sixteen * sixteen * four * two - one;
    
//     (num48271 * state) % num0x7fffffff
// }

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
// pub enum Seed<T> {
//     #[default]
//     OS,
//     Input(T),
// }