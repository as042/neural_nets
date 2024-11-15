use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::autodiff::{real::{real_math::RealMath, Real}, tape::Tape, var::Var};
use crate::rng::{Seed, os_seed, lehmer_rng};

use super::Layout;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize, Encode, Decode)]
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
            init_seed = os_seed();
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
    pub fn var_params<'t>(&self, tape: *const Tape<T>) -> Params<Var<'t, T>> {
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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::autodiff::tape::*;

    #[test]
    fn test_default_params() {
        let layout = Layout::builder()
            .input_layer(1)
            .feed_forward_layer(ActivationFn::ReLU, 2)
            .feed_forward_layer(ActivationFn::Linear, 1)
            .build();
    
        let params = Params::<f64>::default_params(&layout);
        let params2 = Params {
            weights: vec![1.0, 1.0, 1.0, 1.0],
            biases: vec![1.0, 1.0, 1.0],
            others: vec![],
        };
    
        assert_eq!(params, params2);
    }
    
    #[test]
    fn test_random_params() {    
        let layout = Layout::builder()
            .input_layer(1)
            .feed_forward_layer(ActivationFn::ReLU, 1)
            .feed_forward_layer(ActivationFn::Linear, 1)
            .build();
    
        let params = Params::<f64>::random_params(&layout, Seed::Input(1122219.0));
        let params2 = Params {
            weights: vec![-0.9399092877283692, -0.3612265856936574],
            biases: vec![-0.7685036603361368, -0.440182882361114],
            others: vec![],
        };
    
        assert_eq!(params, params2);
    }
    
    #[test]
    fn test_var_params() {    
        let mut tape = Tape::new();
    
        let layout = Layout::builder()
            .input_layer(1)
            .feed_forward_layer(ActivationFn::ReLU, 1)
            .feed_forward_layer(ActivationFn::Linear, 1)
            .build();
    
        let params = Params::<f64>::default_params(&layout);
    
        let var_params = params.var_params(&mut tape);
        let var_params2 = Params {
            weights: vec![tape.new_var(1.0), tape.new_var(1.0)],
            biases: vec![tape.new_var(1.0), tape.new_var(1.0)],
            others: vec!(),
        };
    
        assert_ne!(var_params, var_params2);
        assert_eq!(var_params.weights.iter().map(|x| x.val()).collect::<Vec<f64>>(), var_params2.weights.iter().map(|x| x.val()).collect::<Vec<f64>>());
    }
}