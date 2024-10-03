use crate::autodiff::grad_num::GradNum;
use crate::network::Network;

use super::network_data::NetworkData;
use super::run_settings::RunSettings;

impl<'t, T: GradNum> Network<'t, T> {
    /// Runs `self` with the given input. Currently only works for basic feedforward networks.
    #[inline]
    pub fn run(&self, settings: &RunSettings<T>) {
        let input = &settings.input;

        assert_eq!(input.len(), self.layout().layers()[0].num_neurons()); // the correct number of inputs must be provided

        let mut net_data = NetworkData::new(self.layout().layers(), self.params());

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
            for n in 0..self.nth_layer(l).num_neurons() {
                let neuron_idx = self.nth_layer(l).neuron_start_idx() + n;

                let mut sum = self.neurons[neuron_idx].bias();

                for w in 0..self.prev_layer(l).num_neurons() {
                    sum = sum + self.weights[self.nth_neuron(neuron_idx).weight_start_idx() + w].value() * 
                        self.neurons[self.prev_layer(l).neuron_start_idx() + w].activation();
                }

                self.neurons[neuron_idx].activation = self.nth_layer(l).activation_fn().compute(sum);
            }
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