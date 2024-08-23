use reverse::Var;

use crate::{network::Network, prelude::DiffNetwork};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunSettings {
    pub(crate) input: Vec<f64>,
    pub(crate) clamp: bool,
    pub(crate) print: bool,
}

impl RunSettings {
    /// Creates a new `Self` with the given input and activation function.
    #[inline]
    pub fn new(input: Vec<f64>, clamp: bool) -> Self {
        RunSettings {
            input,
            clamp,
            print: false,
        }
    }

    /// Creates a new `Self` with the given input and activation function and has printing enabled.
    #[inline]
    pub fn new_with_print(input: Vec<f64>, clamp: bool) -> Self {
        RunSettings {
            input,
            clamp,
            print: true,
        }
    }
}

impl Network {
    /// Runs `self` with the given input.
    #[inline]
    pub fn run(&mut self, settings: &RunSettings) {
        let input = &settings.input;

        assert_eq!(input.len(), self.input_layer().num_neurons()); // input vec must be same len as input layer

        // compute first layer
        for n in 0..self.nth_comput_layer(0).num_neurons() {
            let mut sum = self.nth_neuron(n).bias();

            for w in 0..self.input_layer().num_neurons() {
                sum += self.nth_weight(self.nth_neuron(n).weight_start_idx() + w).value() * input[w];
            }

            self.neurons[n].activation = self.nth_comput_layer(0).activation_fn().compute(sum);
        }

        // compute all other layers
        for l in 2..self.num_layers() {
            for n in 0..self.nth_layer(l).num_neurons() {
                let neuron_idx = self.nth_layer(l).neuron_start_idx() + n;

                let mut sum = self.nth_neuron(neuron_idx).bias();

                for w in 0..self.prev_layer(l).num_neurons() {
                    sum += self.nth_weight(self.nth_neuron(neuron_idx).weight_start_idx() + w).value() * 
                        self.nth_neuron(self.prev_layer(l).neuron_start_idx() + w).activation();
                }

                self.neurons[neuron_idx].activation = self.nth_layer(l).activation_fn().compute(sum);
            }
        }

        if settings.print {
            println!("Output: {:?}", self.output());
        }
    }
}



// here be dragons

impl<'a> DiffNetwork<'a> {
    pub(crate) fn diff_run(&mut self, settings: RunSettings) -> Vec<Var<'a>> {
        let input = &settings.input;

        // compute first layer
        for n in 0..self.layers[1].num_neurons() {
            let mut sum = self.neurons[n].bias;

            for w in 0..input.len() {
                sum = sum + self.weights[self.neurons[n].weight_start_idx + w].value * input[w];
            }

            self.neurons[n].activation = self.layers[1].activation_fn.diff_compute(sum);
        }

        // compute all other layers
        for l in 2..self.layers.len() {
            for n in 0..self.layers[l].num_neurons() {
                let neuron_idx = self.layers[l].neuron_start_idx() + n;

                let mut sum = self.neurons[neuron_idx].bias;

                for w in 0..self.layers[l - 1].num_neurons() {
                    sum = sum + self.weights[self.neurons[neuron_idx].weight_start_idx + w].value * 
                        self.neurons[self.layers[l - 1].neuron_start_idx() + w].activation;
                }

                self.neurons[neuron_idx].activation = self.layers[l].activation_fn.diff_compute(sum);
            }
        }

        if settings.print {
            println!("Output: {:?}", self.output());
        }

        self.output()
    }
}