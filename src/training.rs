pub mod clamp_settings;
pub mod cost;
pub mod eta;
pub mod training_results;
pub mod training_settings;

use crate::autodiff::real::operations::OperateWithReal;
use crate::autodiff::real::real_math::RealMath;
use crate::autodiff::real::Real;
use crate::autodiff::tape::Tape;
use crate::network::Network;
use crate::prelude::Params;

use clamp_settings::ClampSettings;
use eta::Eta;
use training_settings::TrainingSettings;

impl Network {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train<'t, T: Real, U: RealMath + OperateWithReal<T>>(&self, settings: &TrainingSettings<'t, T>, mut params: Params<T>) -> Params<T> {
        for _ in 0..settings.num_epochs {
            for _ in 0..settings.num_batches() {
                let mut tape = Tape::new();
                let vars = params.var_params(&mut tape);

                let mut res = self.forward_pass(&settings.input_set()[0], vars);

                let cost = res.cost(settings.cost_fn(), &settings.output_set()[0]);

                let full_gradient = cost.backprop();
                let grad = full_gradient.wrt_inputs();

                params = Self::adjust_params(grad, settings.clamp_settings(), settings.eta(), &params);
            }
        }

        params
    }

    /// Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params<'t, T: Real>(grad: &[T], clamp_settings: &ClampSettings<T>, eta: &Eta<T>, params: &Params<T>) -> Params<T> {
        let weights_len = params.weights().len();
        let mut new_weights = Vec::with_capacity(weights_len);

        let biases_len = params.biases().len();
        let mut new_biases = Vec::with_capacity(biases_len);

        for w in 0..weights_len {
            let weight = params.weights()[w] - eta.val() * grad[w];
            
            let weight = weight.clamp(clamp_settings.weight_min(), clamp_settings.weight_max());

            new_weights.push(weight);
        }
        for b in 0..biases_len {
            let bias = params.biases()[b] - eta.val() * grad[weights_len + b];
            
            let bias = bias.clamp(clamp_settings.bias_min(), clamp_settings.bias_max());

            new_biases.push(bias);
        }

        // others not implemented
        Params::new(new_weights, new_biases, Vec::default())
    }
}