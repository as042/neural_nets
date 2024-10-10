use std::time::SystemTime;

use crate::autodiff::{real::{real_math::RealMath, Real}, tape::Tape, var::Var};

use super::Layout;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Params<U: RealMath> {
    pub(super) weights: Vec<U>,
    pub(super) biases: Vec<U>,
    pub(super) others: Vec<U>, // currently unused
}

impl<U: RealMath> Params<U> {
    #[inline]
    pub fn new(weights: Vec<U>, biases: Vec<U>, others: Vec<U>) -> Self {
        Params { 
            weights, 
            biases, 
            others, 
        }
    }

    #[inline]
    pub fn weights(&self) -> &Vec<U> {
        &self.weights
    }

    #[inline]
    pub fn biases(&self) -> &Vec<U> {
        &self.biases
    }

    #[inline]
    pub fn others(&self) -> &Vec<U> {
        &self.others
    }
}

impl<T: Real> Params<T> {
    #[inline]
    pub fn default_params(layout: &Layout) -> Self {
        let num_weights = layout.num_weights();
        let num_biases = layout.num_biases();

        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..layout.num_weights() {
            weights.push(T::one());
        }

        let mut biases = Vec::with_capacity(num_biases);
        for _ in 0..layout.num_biases() {
            biases.push(T::one());
        }

        Params {
            weights,
            biases,
            others: Vec::default(),
        }
    }

    #[inline]
    pub fn random_params(layout: &Layout, seed: Seed<T>) -> Params<T> { 
        let mut init_seed = T::one();

        if seed == Seed::OS {
            let system_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();

            let two = T::one() + T::one();
            let ten = two * two * two + two;
            let mut val = T::zero();
            for d in system_time.to_string().chars().map(|d| d.to_digit(10).unwrap()).rev().enumerate() {
                let mut ten_power = T::one();
                for _ in 0..d.0 {
                    ten_power = ten_power * ten;
                }
                val = val + ten_power * match d.1 {
                    0 => T::zero(),
                    1 => T::one(),
                    2 => two,
                    3 => two + T::one(),
                    4 => two + two,
                    5 => two + two + T::one(),
                    6 => ten - two - two,
                    7 => ten - two - T::one(),
                    8 => ten - two,
                    9 => ten - T::one(),
                    _ => panic!("Invalid digit"),
                };
            }

            init_seed = val;
        }
        else if let Seed::Input(val) = seed {
            if val <= T::zero() { panic!("Seed must be greater than 0") };

            init_seed = val;
        }

        let mut rng = lehmer_rng(init_seed);

        let two = T::one() + T::one();
        let two_to_the_30 = two.powf(two.powf(two + two + T::one()) - two);

        let num_weights = layout.num_weights();
        let num_biases = layout.num_biases();

        let mut weight_vars = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            rng = lehmer_rng(rng);
            weight_vars.push(rng / two_to_the_30 - T::one());
        }

        let mut bias_vars = Vec::with_capacity(num_biases);
        for _ in 0..num_biases {
            rng = lehmer_rng(rng);
            bias_vars.push(rng / two_to_the_30 - T::one());
        }

        Params { 
            weights: weight_vars, 
            biases: bias_vars, 
            others: Vec::default() 
        }
    }

    #[inline]
    pub fn var_params<'t>(&self, tape: *mut Tape<T>) -> Params<Var<'t, T>> {
        let num_weights = self.weights().len();
        let num_biases = self.biases().len();

        let mut weights = Vec::with_capacity(num_weights);
        for w in 0..num_weights {
            unsafe {
                weights.push(tape.as_ref().unwrap().new_var(self.weights()[w]));
            }
            
        }

        let mut biases = Vec::with_capacity(num_biases);
        for b in 0..num_biases {
            unsafe {
                biases.push(tape.as_ref().unwrap().new_var(self.biases()[b]));
            }
        }

        Params {
            weights,
            biases,
            others: Vec::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Seed<T> {
    #[default]
    OS,
    Input(T),
}

#[inline]
fn lehmer_rng<T: Real>(state: T) -> T {
    let one = T::one();
    let two = T::one() + T::one();
    let three = two + one;
    let four = three + one;
    let sixteen = four * four;

    // 4^7 * 3 - (4^5) + (4 * 4 * 3 * 3) - 1
    let num48271 = sixteen * sixteen * sixteen * four * three - sixteen * sixteen * four + sixteen * three * three - one;
    let num0x7fffffff = sixteen * sixteen * sixteen * sixteen * sixteen * sixteen * sixteen * four * two - one;
    
    (num48271 * state) % num0x7fffffff
}