pub mod clamp_settings;
pub mod cost;
pub mod eta;
pub mod training_results;
pub mod training_settings;

// use crate::autodiff::grad_num::GradNum;
// use crate::network::Network;
// use crate::prelude::ParamHelper;

// use eta::Eta;
// use training_results::TrainingResults;
// use training_settings::TrainingSettings;

// impl<'t, T: GradNum> Network<'t, T> {    
//     /// Runs `self` with the given input and adjusts params to minimize cost.
//     #[inline]
//     pub fn train(&mut self, settings: &TrainingSettings<'t, T>, param_helper: &ParamHelper<T>) -> TrainingResults<T> {
//         let res = self.run(&settings.input_set()[0]);

//         let cost = res.cost(settings.cost_fn(), &settings.output_set()[0]);

//         let full_gradient = cost.backprop();
//         let grad = full_gradient.wrt_inputs();

//         self.adjust_params(grad, settings.eta());

//         TrainingResults {
//             grad: grad.to_vec(),
//             output: self.output(),
//             cost,
//         }
//     }

//     /// Adjusts weights and biases according to grad.
//     #[inline]
//     fn adjust_params(&mut self, grad: &[T], eta: Eta<T>, clamp: bool) {
//         let weights_len = self.params().weights().len();
//         for w in 0..weights_len {
//             self.weights[w].value = self.weights[w].value - eta * grad[w];
//             if clamp { self.weights[w].value = self.weights[w].value.clamp(-1.0, 1.0); }
//         }
//         for b in 0..self.neurons.len() {
//             self.neurons[b].bias = self.neurons[b].bias - eta * grad[b + weights_len];
//             if clamp { self.neurons[b].bias = self.neurons[b].bias.clamp(-1.0, 1.0); }
//         }
//     }
// }