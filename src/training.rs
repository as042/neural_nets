pub mod clamp_settings;
pub mod cost;
pub mod data_set;
pub mod eta;
pub mod trainer;
pub mod training_results;
pub mod training_settings;

use crate::autodiff::{real::{operations::OperateWithReal, real_math::RealMath, Real}, tape::Tape, var::Var};
use crate::network::{Network, params::Params};
use crate::rng::shuffle;

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
                params = self.per_batch(settings, &samples, &params, e, b);
            }
        }

        params
    }

    #[inline]
    fn per_batch<'t, T: Real>(&self, settings: &TrainingSettings<'t, T>, samples: &Vec<usize>, params: &Params<T>, epoch: usize, batch: usize) -> Params<T> {
        let mut tape = Tape::new();
        let vars = params.var_params(&mut tape);

        let costs = self.get_costs(settings, samples, &vars, batch);

        // combine costs before backprop
        let mut total_cost = costs[0];
        for cost in costs[1..].iter() {
            total_cost = total_cost + *cost;
        }
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        let avg_cost = total_cost / (ten * ten * ten * ten);

        let full_gradient = avg_cost.backprop();
        let grad = full_gradient.wrt_inputs();

        Self::adjust_params(grad, settings.clamp_settings(), settings.eta(), epoch, &params)
    }

    #[inline]
    fn get_costs<'t, T: Real>(&self, settings: &TrainingSettings<'t, T>, samples: &Vec<usize>, vars: &Params<Var<'t, T>>, batch: usize) -> Vec<Var<'t, T>> {
        let mut costs = Vec::with_capacity(settings.num_epochs);
        for s in 0..settings.batch_size {
            let sample_idx = samples[batch + s];
            let mut res = self.forward_pass(&settings.data_set().nth_input(sample_idx).to_vec(), vars);
    
            let cost = res.cost(settings.cost_fn(), &settings.data_set.nth_output(sample_idx).to_vec());
            costs.push(cost);
        }

        costs
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

#[cfg(test)]
mod tests {
    use crate::{autodiff::tape::Tape, prelude::*};

    #[test]
    fn test_get_costs() {
        let layout = Layout::builder()
            .input_layer(2)
            .feed_forward_layer(ActivationFn::SiLU, 2)
            .feed_forward_layer(ActivationFn::Linear, 2)
            .build();

        let net = Network::new(layout);

        let data_set = DataSet::builder()
            .sample(vec![0.2, 0.1], vec![-0.3, 0.6])
            .sample(vec![-0.82, 0.8], vec![0.6, 0.4])
            .sample(vec![0.27, 0.55], vec![-0.21, -0.5])
            .sample(vec![0.235, -0.34], vec![0.8, 0.37])
            .sample(vec![-0.9, 0.27], vec![0.5, -0.6])
            .sample(vec![-0.1, -0.8], vec![-0.74, 0.25])
            .build();

        let params = net.random_params(Seed::Input(100.0));

        let settings = TrainingSettings {
            batch_size: 3,
            num_epochs: 2,
            cost_fn: CostFn::MAE,
            clamp_settings: ClampSettings::NO_CLAMP,
            eta: Eta::point_one(),
            data_set: data_set,
            stoch_shuffle_seed: Seed::Input(100.0),
        };

        let samples = vec![1, 3, 0, 5, 4, 2];
        let mut tape = Tape::new();
        let vars = params.var_params(&mut tape);

        let costs = net.get_costs(&settings, &samples, &vars, 0);
    }
}