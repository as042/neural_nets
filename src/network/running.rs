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

// #[test]
// fn test_run() {
//     let mut builder = Network::new();
//     let mut net: Network<f64> = builder
//         .add_layer(Layer::new_input().add_neurons(2))
//         .add_layer(Layer::new_comput().add_neurons(2).add_activation_fn(ActivationFn::Sigmoid))
//         .add_layer(Layer::new_comput().add_neurons(2).add_activation_fn(ActivationFn::Sigmoid))
//         .build();

//     net.randomize_params(Some(0));

//     let settings = &RunSettings::new(
//         vec![-0.2, 0.1], 
//         false
//     );

//     net.run(settings);

//     assert_eq!(net.output(), vec![0.9384751282963776, 0.9156491958141794]);
// }