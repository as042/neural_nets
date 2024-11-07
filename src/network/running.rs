use crate::autodiff::real::operations::OperateWithReal;
use crate::autodiff::real::real_math::RealMath;
use crate::autodiff::real::Real;
use crate::network::Network;

use super::network_data::NetworkData;
use super::params::Params;
use super::run_results::RunResults;

impl Network {
    #[inline]
    pub fn run<T: Real + OperateWithReal<T>>(&self, input: &Vec<T>, params: &Params<T>) -> RunResults<T, T> {
        self.forward_pass(input, params)
    }

    /// Runs `self` with the given input. Currently only works for basic feedforward networks.
    #[inline]
    pub(crate) fn forward_pass<T: Real, U: RealMath + OperateWithReal<T>>(&self, input: &Vec<T>, params: &Params<U>) -> RunResults<T, U> {
        assert_eq!(input.len(), self.layout().layers()[0].num_neurons()); // the correct number of inputs must be provided

        let mut net_data = NetworkData::new(self.layout().layers(), params);

        // compute first layer
        for n in 0..net_data.layer_data[0].layer.num_neurons() {
            let mut sum = net_data.neuron_data[n].bias;

            for w in 0..input.len() {
                sum = sum + net_data.weight_data[net_data.neuron_data[n].weight_start_idx + w] * input[w];
            }

            net_data.neuron_data[n].activation = Some(net_data.layer_data[0].layer.activation_fn().compute(sum));
        }

        // compute all other layers
        for l in 1..net_data.layer_data.len() {
            for n in 0..net_data.layer_data[l].layer.num_neurons() {
                let neuron_idx = net_data.layer_data[l].neuron_start_idx + n;

                let mut sum = net_data.neuron_data[neuron_idx].bias;

                for w in 0..net_data.layer_data[l - 1].layer.num_neurons() {
                    sum = sum + net_data.weight_data[net_data.neuron_data[neuron_idx].weight_start_idx + w] * 
                        net_data.neuron_data[net_data.layer_data[l - 1].neuron_start_idx + w].activation.unwrap();
                }

                net_data.neuron_data[neuron_idx].activation = Some(net_data.layer_data[l].layer.activation_fn().compute(sum));
            }
        }

        RunResults { 
            output: net_data.output(),
            _marker: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_run() {
        let net = Network::builder()
            .input_layer(2)
            .feed_forward_layer(ActivationFn::ReLU, 2)
            .feed_forward_layer(ActivationFn::Linear, 2)
            .build();
    
        let input = vec![0.5, 0.1];
        let params = net.default_params();
    
        let res = net.run(&input, &params);
    
        assert_eq!(res.output(), &[4.2, 4.2]);
    }
}