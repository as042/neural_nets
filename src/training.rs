pub mod clamp_settings;
pub mod cost;
pub mod data_set;
pub mod eta;
pub mod trainer;
pub mod training_results;
pub mod training_settings;

use crate::autodiff::{real::{operations::OperateWithReal, Real, real_math::RealMath}, tape::Tape};
use crate::network::{Network, params::Params};
use crate::rng::{shuffle, Seed};

use clamp_settings::ClampSettings;
use eta::Eta;
use training_settings::TrainingSettings;

impl Network {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train<'t, T: Real, U: RealMath + OperateWithReal<T>>(&self, settings: &TrainingSettings<'t, T>, mut params: Params<T>) -> Params<T> {
        for e in 0..settings.num_epochs {
            let mut samples: Vec<usize> = (0..settings.data_set().len()).collect();
            shuffle(&mut samples, settings.stoch_shuffle_seed);

            for b in 0..settings.num_batches() {
                let mut tape = Tape::new();
                let mut costs = Vec::with_capacity(settings.num_epochs);
                for s in 0..*settings.batch_size() {
                    let vars = params.var_params(&mut tape);
                    let mut res = self.forward_pass(&settings.data_set().nth_input(b + s).to_vec(), &vars);

                    let cost = res.cost(settings.cost_fn(), &settings.data_set.nth_output(b + s).to_vec());
                    costs.push(cost);
                }

                // combine costs before backprop
                let mut total_cost = costs[0];
                for cost in costs[1..].iter() {
                    total_cost = total_cost + *cost;
                }

                let full_gradient = total_cost.backprop();
                let grad = full_gradient.wrt_inputs();

                params = Self::adjust_params(grad, settings.clamp_settings(), settings.eta(), e, &params);
            }
        }

        params
    }

    /// Adjusts weights and biases according to grad. KNOWN PROBLEM: Large eta value
    #[inline]
    fn adjust_params<'t, T: Real>(grad: &[T], clamp_settings: &ClampSettings<T>, eta: &Eta<T>, epoch: usize, params: &Params<T>) -> Params<T> {
        let weights_len = params.weights().len();
        let mut new_weights = Vec::with_capacity(weights_len);

        let biases_len = params.biases().len();
        let mut new_biases = Vec::with_capacity(biases_len);

        for w in 0..weights_len {
            let weight = params.weights()[w] - eta.val(epoch) * grad[w];
            let weight = weight.clamp(clamp_settings.weight_min(), clamp_settings.weight_max());

            new_weights.push(weight);
        }
        for b in 0..biases_len {
            let bias = params.biases()[b] - eta.val(epoch) * grad[weights_len + b];
            let bias = bias.clamp(clamp_settings.bias_min(), clamp_settings.bias_max());

            new_biases.push(bias);
        }

        // others not implemented
        Params::new(new_weights, new_biases, Vec::default())
    }
}