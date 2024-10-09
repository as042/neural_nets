pub mod clamp_settings;
pub mod cost;
pub mod eta;
pub mod training_results;
pub mod training_settings;

use crate::autodiff::real::Real;
use crate::network::Network;
use crate::prelude::{TapeContainer, Params};

use clamp_settings::ClampSettings;
use eta::Eta;
use training_results::TrainingResults;
use training_settings::TrainingSettings;

impl Network {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train<'t, T: Real>(&mut self, settings: &TrainingSettings<'t, T>, params: &Params<'t, T>, param_helper: &'t mut TapeContainer<T>) -> () {
        let mut res = self.forward_pass(&settings.input_set()[0], params);

        let cost = res.cost(settings.cost_fn(), &settings.output_set()[0]);

        let full_gradient = cost.backprop();
        let grad = full_gradient.wrt_inputs();

        let params = self.adjust_params(grad, settings.clamp_settings(), settings.eta(), params, param_helper);

        ()
    }

    /// Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params<'t, T: Real>(&self, grad: &[T], clamp_settings: &ClampSettings<T>, eta: &Eta<T>, params: &Params<'t, T>, param_helper: &'t mut TapeContainer<T>) -> Params<'t, T> {
        let weights_len = params.weights().len();
        let mut new_weights = Vec::with_capacity(weights_len);

        let biases_len = params.biases().len();
        let mut new_biases = Vec::with_capacity(biases_len);

        for w in 0..weights_len {
            let mut weight = params.weights()[w].val() - eta.val() * grad[w];
            
            if weight < clamp_settings.weight_min() {
                weight = clamp_settings.weight_min();
            }
            if weight > clamp_settings.weight_max() {
                weight = clamp_settings.weight_max();
            }

            new_weights.push(weight);
        }
        for b in 0..biases_len {
            let mut bias = params.biases()[b].val() - eta.val() * grad[weights_len + b];
            
            if bias < clamp_settings.bias_min() {
                bias = clamp_settings.bias_min();
            }
            if bias > clamp_settings.bias_max() {
                bias = clamp_settings.bias_max();
            }

            new_biases.push(bias);
        }

        // others not implemented
        param_helper.params(new_weights, new_biases, Vec::default())
    }
}