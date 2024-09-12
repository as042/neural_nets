use crate::prelude::*;

/// Used to configure how a `Network` is run.
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct RunSettings<T: GradNum> {
    pub(crate) input: Vec<T>,
    pub(crate) clamp: bool,
    pub(crate) print: bool,
}

impl<T: GradNum> RunSettings<T> {
    /// Creates a new `Self` with the given input and activation function.
    #[inline]
    pub fn new(input: Vec<T>, clamp: bool) -> Self {
        RunSettings {
            input,
            clamp,
            print: false,
        }
    }

    /// Creates a new `Self` with the given input and activation function and has printing enabled.
    #[inline]
    pub fn new_with_print(input: Vec<T>, clamp: bool) -> Self {
        RunSettings {
            input,
            clamp,
            print: true,
        }
    }
}

impl<'t, T: GradNum> Network<'t, T> {
    /// Runs `self` with the given input.
    #[inline]
    pub fn run(&mut self, settings: &RunSettings<T>) {
        let input = &settings.input;

        assert_eq!(input.len(), self.input_layer().num_neurons()); // input vec must be same len as input layer

        // compute first layer
        for n in 0..self.nth_comput_layer(0).num_neurons() {
            let mut sum = self.nth_neuron(n).bias();

            for w in 0..self.input_layer().num_neurons() {
                sum = sum + self.nth_weight(self.nth_neuron(n).weight_start_idx() + w).value() * input[w];
            }

            self.neurons[n].activation = self.nth_comput_layer(0).activation_fn().compute(sum);
        }

        // compute all other layers
        for l in 2..self.num_layers() {
            for n in 0..self.nth_layer(l).num_neurons() {
                let neuron_idx = self.nth_layer(l).neuron_start_idx() + n;

                let mut sum = self.nth_neuron(neuron_idx).bias();

                for w in 0..self.prev_layer(l).num_neurons() {
                    sum = sum + self.nth_weight(self.nth_neuron(neuron_idx).weight_start_idx() + w).value() * 
                        self.nth_neuron(self.prev_layer(l).neuron_start_idx() + w).activation();
                }

                self.neurons[neuron_idx].activation = self.nth_layer(l).activation_fn().compute(sum);
            }
        }
    }
}

#[test]
fn test_run() {
    let mut builder = Network::new();
    let mut net: Network<f64> = builder
        .add_layer(Layer::new_input().add_neurons(2))
        .add_layer(Layer::new_comput().add_neurons(2).add_activation_fn(ActivationFn::Sigmoid))
        .add_layer(Layer::new_comput().add_neurons(2).add_activation_fn(ActivationFn::Sigmoid))
        .build();

    net.randomize_params(Some(0));

    let settings = &RunSettings::new(
        vec![-0.2, 0.1], 
        false
    );

    net.run(settings);

    assert_eq!(net.output(), vec![0.9384751282963776, 0.9156491958141794]);
}