use num_traits::real::Real;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{layer::*, network_builder::NetworkBuilder, neuron::Neuron, prelude::Var, weight::Weight};

pub trait GradNum: Real + Default {}

impl<T: Real + Default> GradNum for T {}

/// Used to create, run, and train neural networks.
/// # Examples
/// ```
/// use neural_nets::prelude::*;
/// 
/// // create the network
/// let mut net = Network::new()
///     .add_layer(Layer::new_input().add_neurons(1))
///     .add_layer(Layer::new_comput().add_neurons(3).add_activation_fn(ActivationFn::Sigmoid))
///     .add_layer(Layer::new_comput().add_neurons(3).add_activation_fn(ActivationFn::GELU))
///     .build();
///
/// // give it random parameters
/// net.randomize_params(None);
///
/// // indicate the settings with which to run the network
/// let settings = &RunSettings::new(
///     vec![0.2], 
///     true
/// );
/// let desired_output = vec![0.5, 0.7, 0.56];
///
/// // save the initial cost of the network to compare with the final cost
/// net.run(settings);
/// let init_cost = net.total_cost(&desired_output);
///
/// // train the network 1000 times and print the cost after each time
/// for _ in 0..1000 {
///     println!("cost: {:?}", net.train(settings, &vec![0.5, 0.7, 0.56], 0.1).cost());
/// }    
/// 
/// println!("{init_cost}");
/// ```
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Network<'t, T: GradNum> {
    pub(crate) layers: Vec<Layer>,
    pub(crate) neurons: Vec<Neuron<'t, T>>,
    pub(crate) weights: Vec<Weight<'t, T>>,
}

// useless type just there so it compiles
impl<'t> Network<'t, f32> {
    /// Creates a builder to aid in construction of a `Network`.
    /// # Examples
    /// ```
    /// let mut net = Network::new()
    ///     .add_layer(Layer::new_input().add_neurons(1))
    ///     .add_layer(Layer::new_comput().add_neurons(3))
    ///     .add_layer(Layer::new_comput().add_neurons(3))
    ///     .build();
    /// ```
    #[inline]
    pub fn new() -> NetworkBuilder {
        NetworkBuilder::new()
    }
}

impl<'t, T: GradNum> Network<'t, T> {
    /// Returns the input `Layer`.
    #[inline]
    pub fn input_layer(&self) -> &Layer {
        &self.layers[0]
    }

    /// Returns the `Layer`s.
    #[inline]
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    /// Returns the `Neuron`s.
    #[inline]
    pub fn neurons(&self) -> &Vec<Neuron<T>> {
        &self.neurons
    }

    /// Returns the 'Weight's.
    #[inline]
    pub fn weights(&self) -> &Vec<Weight<T>> {
        &self.weights
    }

    /// Returns the number of `Layer`s.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of `Neuron`s, not counting input.
    #[inline]
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of `Weight`s.
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

    /// Returns the next `Layer`. Panics if idx >= idx of last `Layer`.
    #[inline]
    pub fn next_layer(&self, idx: usize) -> &Layer {
        assert!(idx < self.num_layers() - 1);
        &self.layers[idx + 1]
    }

    /// Returns the last `Layer`.
    #[inline]
    pub fn last_layer(&self) -> &Layer {
        self.layers.last().unwrap()
    }

    /// Returns the nth `Neuron`.
    #[inline]
    pub fn nth_neuron(&self, idx: usize) -> &Neuron<T> {
        &self.neurons[idx]
    }

    /// Returns the nth `Weight`.
    #[inline]
    pub fn nth_weight(&self, idx: usize) -> &Weight<T> {
        &self.weights[idx]
    }

    /// Returns the output.
    #[inline]
    pub fn output(&self) -> Vec<T> {
        let mut vec = Vec::default();

        for n in 0..self.last_layer().num_neurons {
            vec.push(self.neurons[self.last_layer().neuron_start_idx + n].activation)
        }

        vec
    }

    /// Sets the weights and biases of a specific `Neuron`.
    #[inline]
    pub fn set_neuron_params(&mut self, neuron_idx: usize, bias: T, weights: Vec<T>) {
        let num_weights = self.nth_neuron(neuron_idx).num_weights();
        let weight_start_idx = self.nth_neuron(neuron_idx).weight_start_idx();
        assert_eq!(weights.len(), num_weights);

        self.neurons[neuron_idx].bias = bias;

        for w in 0..num_weights {
            self.weights[weight_start_idx + w].value = weights[w];
        }
    }
}

impl<'t, T: GradNum + From<f64>> Network<'t, T> {
    /// Randomizes all weights and biases.
    /// # Examples
    /// ```
    /// // create the network
    /// let mut net = Network::new()
    ///     .add_layer(Layer::new_input().add_neurons(1))
    ///     .add_layer(Layer::new_comput().add_neurons(1))
    ///     .build();
    ///
    /// // give it random parameters
    /// net.randomize_params(None);
    /// 
    /// // give it seeded random parameters 
    /// net.randomize_params(Some(5334));
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
}

#[test]
fn test_set_neuron_params() {
    let mut net: Network<f64> = Network::new()
        .add_layer(Layer::new_input().add_neurons(1))
        .add_layer(Layer::new_comput().add_neurons(1))
        .build();

    net.set_neuron_params(0, 0.1, vec![0.3]);

    assert_eq!(net, 
        Network { 
            layers: vec![Layer { num_neurons: 1, ..Default::default() }, Layer { num_neurons: 1, layer_type: LayerType::Comput, ..Default::default() }], 
            neurons: vec![Neuron { bias: 0.1, num_weights: 1, ..Default::default() }], 
            weights: vec![Weight { value: 0.3 }],
        }
    );
}

#[test]
fn test_randomize_params() {
    let mut net: Network<f64> = Network::new()
        .add_layer(Layer::new_input().add_neurons(1))
        .add_layer(Layer::new_comput().add_neurons(1))
        .build();

    net.randomize_params(Some(0));

    assert_eq!(net, 
        Network { 
            layers: vec![Layer { num_neurons: 1, ..Default::default() }, Layer { num_neurons: 1, layer_type: LayerType::Comput, ..Default::default() }], 
            neurons: vec![Neuron { bias: -0.1363131108415594, num_weights: 1, ..Default::default() }], 
            weights: vec![Weight { value: 0.8363016617062469 }],
        }
    );
} 

#[test]
fn simple_network_test() {
    use crate::prelude::*;

    let mut net: Network<f64> = Network::new()
        .add_layer(Layer::new_input().add_neurons(3))
        .add_layer(Layer::new_comput().add_neurons(3).add_activation_fn(ActivationFn::Sigmoid))
        .add_layer(Layer::new_comput().add_neurons(2).add_activation_fn(ActivationFn::Sigmoid))
        .build();

    net.randomize_params(Some(1));

    let settings = &RunSettings::new(
        vec![0.21, 0.1, -0.39], 
        false
    );

    for _ in 0..10000 {
        net.train(settings, &vec![0.59, 0.7], 0.1);
    }

    assert_eq!(net.train(settings, &vec![0.59, 0.7], 0.1).cost(), 4.930380657631324e-32); // was 4.5606021083089745e-31 in an earlier build
}

#[test]
fn identity_test() {
    use crate::prelude::*;

    let mut net: Network<f64> = Network::new()
        .add_layer(Layer::new_input().add_neurons(1))
        .add_layers(3, Layer::new_comput().add_neurons(50).add_activation_fn(ActivationFn::Sigmoid))
        .add_layer(Layer::new_comput().add_neurons(1).add_activation_fn(ActivationFn::Sigmoid))
        .build();

    net.randomize_params(Some(0));

    let mut avg_cost = 0.0;

    for i in 0..100 {
        avg_cost += net.train(&RunSettings::new(vec![i as f64 / 100.0], true), &vec![i as f64 / 1000.0], 0.1).cost();
    }

    avg_cost /= 100.0;

    panic!();
    // assert_eq!((avg_cost * 1E6).round() / 1E6, 0.108389);
}