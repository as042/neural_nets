// use crate::{layer::LayerType, neuron::Neuron, prelude::{ActivationFn, GradNum, Layer, Network}, weight::Weight};

// /// Assists in the construction of a `Network`.
// #[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
// pub struct NetworkBuilder {
//     layer_builders: Vec<LayerBuilder>,
// }

// /// Assists in the construction of a `Network`.
// #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
// pub struct LayerBuilder {
//     num_neurons: usize,
//     layer_type: LayerType,
//     activation_fn: ActivationFn,
// }

// impl NetworkBuilder {
//     /// Creates a new `NetworkBuilder`.
//     #[inline]
//     pub(crate) fn new() -> Self {
//         Self::default()
//     }

//     /// Adds a `Layer` to the network.
//     #[inline]
//     pub fn add_layer(&mut self, layer_builder: &LayerBuilder) -> &mut Self {
//         self.layer_builders.push(*layer_builder);
//         self
//     }

//     /// Adds a `Layer` to the network multiple times.
//     #[inline]
//     pub fn add_layers(&mut self, num: usize, layer_builder: &LayerBuilder) -> &mut Self {
//         for _ in 0..num {
//             self.layer_builders.push(*layer_builder);
//         }
//         self
//     }

//     /// Builds the final `Network`.
//     #[inline]
//     pub fn build<T: GradNum>(&self) -> Network<T> {
//         assert!(self.layer_builders.len() > 1); // there must be at least 2 layers
//         assert_eq!(self.layer_builders[0].layer_type, LayerType::Input); // the first layer must be input
//         assert_eq!(self.layer_builders[1..].iter().find(|x| x.layer_type == LayerType::Input), None); // no other lays can be input

//         let mut net = Network::default();

//         // create the input layer
//         net.layers = vec![Layer { num_neurons: self.layer_builders[0].num_neurons, ..Default::default() }];

//         let mut neurons = 0;
//         let mut weights = 0;
//         for l in 1..self.layer_builders.len() {
//             let neurons_in_layer = self.layer_builders[l].num_neurons;
//             let layer_type = self.layer_builders[l].layer_type;
//             let activation_fn = self.layer_builders[l].activation_fn;
//             net.layers.push(Layer { num_neurons: neurons_in_layer, neuron_start_idx: neurons, layer_type, activation_fn });

//             let weights_per_neuron = self.layer_builders[l - 1].num_neurons;
//             for _ in 0..neurons_in_layer {
//                 net.neurons.push(Neuron { num_weights: weights_per_neuron, weight_start_idx: weights, ..Default::default() });
//                 weights += weights_per_neuron;
//             }

//             neurons += neurons_in_layer;
//         }

//         net.weights = vec![Weight::default(); weights];

//         net
//     }
// }

// impl LayerBuilder {
//     /// Creates a new `LayerBuilder` representing an input layer. 
//     /// There can only be one input layer, and it must be the first layer.
//     #[inline]
//     pub(crate) fn new_input() -> Self {
//         Self::default()
//     }

//     /// Creates a new `LayerBuilder` representing a computational layer (hidden or output).
//     /// The first layer cannot be a comput layer.
//     #[inline]
//     pub(crate) fn new_comput() -> Self {
//         Self { layer_type: LayerType::Comput, ..Default::default() }
//     }

//     /// Adds neurons to the layer.
//     #[inline]
//     pub fn add_neurons(&mut self, neurons: usize) -> &mut Self {
//         self.num_neurons = neurons;
//         self
//     }

//     /// Adds an activation fn to the layer.
//     #[inline]
//     pub fn add_activation_fn(&mut self, activation_fn: ActivationFn) -> &mut Self {
//         self.activation_fn = activation_fn;
//         self
//     }
// }