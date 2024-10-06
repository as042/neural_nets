pub mod clamp_settings;
pub mod cost;
pub mod eta;
pub mod training_results;
pub mod training_settings;

use crate::autodiff::grad_num::GradNum;
use crate::network::Network;
use crate::prelude::{ParamHelper, Params};

use clamp_settings::ClampSettings;
use eta::Eta;
use training_results::TrainingResults;
use training_settings::TrainingSettings;

impl<'t, T: GradNum> Network<'t, T> {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, settings: &TrainingSettings<'t, T>, param_helper: &'t mut ParamHelper<T>) -> () {
        let mut res = self.run(&settings.input_set()[0]);

        let cost = res.cost(settings.cost_fn(), &settings.output_set()[0]);

        let full_gradient = cost.backprop();
        let grad = full_gradient.wrt_inputs();

        let params = self.adjust_params(grad, settings.clamp_settings(), settings.eta(), param_helper);

        self.params = params;

        ()
    }

    /// Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params(&self, grad: &[T], clamp_settings: &ClampSettings<T>, eta: &Eta<T>, param_helper: &'t mut ParamHelper<T>) -> Params<'t, T> {
        let weights_len = self.params().weights().len();
        let mut new_weights = Vec::with_capacity(weights_len);

        let biases_len = self.params().biases().len();
        let mut new_biases = Vec::with_capacity(biases_len);

        for w in 0..weights_len {
            let mut weight = self.params().weights()[w].val() - eta.val() * grad[w];
            
            if weight < clamp_settings.weight_min() {
                weight = clamp_settings.weight_min();
            }
            if weight > clamp_settings.weight_max() {
                weight = clamp_settings.weight_max();
            }

            new_weights.push(weight);
        }
        for b in 0..biases_len {
            let mut bias = self.params().biases()[b].val() - eta.val() * grad[weights_len + b];
            
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