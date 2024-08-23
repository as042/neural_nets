use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{layer::*, network_builder::NetworkBuilder, neuron::Neuron, weight::Weight};

/// A network object. `layers` refers to non-input layers.
#[derive(Clone, Debug, Default)]
pub struct Network {
    pub(crate) layers: Vec<Layer>,
    pub(crate) neurons: Vec<Neuron>,
    pub(crate) weights: Vec<Weight>,
}

impl Network {
    /// Creates a builder helper.
    /// # Examples
    /// ```
    /// let mut net = Network::new()
    ///     .add_layer(Layer::new().add_neurons(1))
    ///     .add_layer(Layer::new().add_neurons(3))
    ///     .add_layer(Layer::new().add_neurons(5))
    ///     .add_layer(Layer::new().add_neurons(9))
    ///     .add_layer(Layer::new().add_neurons(3))
    ///     .build();
    ///
    /// net.randomize_params(None);
    ///
    /// let settings = &RunSettings::new(
    ///     vec![0.2], 
    ///     ActivationFunction::Tanh,
    ///     true
    /// );
    /// let desired_output = vec![0.5, 0.7, 0.56];
    ///
    /// net.run(settings);
    /// let init_cost = net.total_cost(&desired_output.iter().map(|f| F::new(*f, 0.0)).collect());
    ///
    /// for _ in 0..1000 {
    ///     println!("{:?}", net.train(settings, 
    ///         &vec![0.5, 0.7, 0.56], 
    ///         0.1).cost()
    ///     );
    /// }
    ///
    /// println!("{init_cost}");
    /// ```
    #[inline]
    pub fn new() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    /// Returns the input layer of `self`.
    #[inline]
    pub fn input_layer(&self) -> &Layer {
        &self.layers[0]
    }

    /// Returns all layers of `self`.
    #[inline]
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    /// Returns the neurons of `self`.
    #[inline]
    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    /// Returns the weights of `self`.
    #[inline]
    pub fn weights(&self) -> &Vec<Weight> {
        &self.weights
    }

    /// Returns the number of layers of `self`.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of neurons of `self`, not counting input.
    #[inline]
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of weights of `self`.
    #[inline]
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Returns the nth `Layer`.
    #[inline]
    pub fn nth_layer(&self, idx: usize) -> &Layer {
        &self.layers[idx]
    }

    /// Returns the nth comput `Layer`.
    #[inline]
    pub fn nth_comput_layer(&self, idx: usize) -> &Layer {
        &self.layers[idx + 1]
    }

    /// Returns the previous `Layer`. Panics if idx = 0.
    #[inline]
    pub fn prev_layer(&self, idx: usize) -> &Layer {
        assert_ne!(idx, 0);
        &self.layers[idx - 1]
    }

    /// Returns the next `Layer`. Panics if idx >= idx of last layer.
    #[inline]
    pub fn next_layer(&self, idx: usize) -> &Layer {
        assert!(idx < self.num_layers() - 1);
        &self.layers[idx + 1]
    }

    /// Returns the last layer of `self`.
    #[inline]
    pub fn last_layer(&self) -> &Layer {
        self.layers.last().unwrap()
    }

    /// Returns the nth `Neuron`.
    #[inline]
    pub fn nth_neuron(&self, idx: usize) -> &Neuron {
        &self.neurons[idx]
    }

    /// Returns the nth `Weight`.
    #[inline]
    pub fn nth_weight(&self, idx: usize) -> &Weight {
        &self.weights[idx]
    }

    /// Returns the output of `self`.
    #[inline]
    pub fn output(&self) -> Vec<f64> {
        let mut vec = Vec::default();

        for n in 0..self.last_layer().num_neurons {
            vec.push(self.neurons[self.last_layer().neuron_start_idx + n].activation)
        }

        vec
    }

    /// Randomizes all weights and biases of `self`.
    #[inline]
    pub fn randomize_params(&mut self, seed: Option<u64>) {
        let mut rng;
        if seed.is_some() {
            rng = ChaCha8Rng::seed_from_u64(seed.unwrap());
        }
        else {
            let mut thread_rng = thread_rng();
            rng = ChaCha8Rng::seed_from_u64(thread_rng.gen());
        }

        for w in 0..self.num_weights() {
            self.weights[w].value = rng.gen_range(-2.0 - 0.01 * w as f64..2.0 + 0.01 * w as f64).into();
        }
        for b in 0..self.num_neurons() {
            self.neurons[b].bias = rng.gen_range(-2.0 - 0.02 * b as f64..2.0 + 0.02 * b as f64).into();
        }
    }

    /// Sets the weights and biases of a specific neuron.
    #[inline]
    pub fn set_neuron_params(&mut self, neuron_idx: usize, bias: f64, weights: Vec<f64>) {
        let num_weights = self.nth_neuron(neuron_idx).num_weights();
        let weight_start_idx = self.nth_neuron(neuron_idx).weight_start_idx();
        assert_eq!(weights.len(), num_weights);

        self.neurons[neuron_idx].bias = bias;

        for w in 0..num_weights {
            self.weights[weight_start_idx + w].value = weights[w];
        }
    }
}

// #[test]
// fn test_set_neuron_params() {
//     let mut net = Network::new()
//         .add_layer(Layer::new_input().add_neurons(1))
//         .add_layer(Layer::new_comput().add_neurons(1))
//         .build();

//     net.set_neuron_params(0, 0.1, vec![0.3]);

//     assert_eq!(net, 
//         Network { 
//             layers: vec![Layer { num_neurons: 1, ..Default::default() }, Layer { num_neurons: 1, layer_type: LayerType::Comput, ..Default::default() }], 
//             neurons: vec![Neuron { bias: 0.1, num_weights: 1, ..Default::default() }], 
//             weights: vec![Weight { value: 0.3 }],
//         }
//     );
// }

// #[test]
// fn test_randomize_params() {
//     let mut net = Network::new()
//         .add_layer(Layer::new_input().add_neurons(1))
//         .add_layer(Layer::new_comput().add_neurons(1))
//         .build();

//     net.randomize_params(Some(0));

//     assert_eq!(net, 
//         Network { 
//             layers: vec![Layer { num_neurons: 1, ..Default::default() }, Layer { num_neurons: 1, layer_type: LayerType::Comput, ..Default::default() }], 
//             neurons: vec![Neuron { bias: -0.1363131108415594, num_weights: 1, ..Default::default() }], 
//             weights: vec![Weight { value: 0.8363016617062469 }],
//         }
//     );
// } 