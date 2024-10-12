use crate::autodiff::real::real_math::RealMath;

use super::{layer::{Layer, LayerType}, params::Params};

#[derive(Clone, Default, Debug, PartialEq, PartialOrd)]
pub(super) struct NetworkData<U: RealMath> {
    pub(super) layer_data: Vec<LayerData>,
    pub(super) neuron_data: Vec<NeuronData<U>>,
    pub(super) weight_data: Vec<U>,
}

impl<'t, U: RealMath> NetworkData<U> {
    #[inline]
    pub(super) fn new(layers: &Vec<Layer>, params: &Params<U>) -> Self {
        assert_eq!(layers[0].layer_type(), LayerType::Input); // first layer is input
        assert!(layers.len() > 1); // more than one layer
        assert!(!layers[1..].iter().any(|&x| x.layer_type != LayerType::FeedForward)); // all but the first layer are feed forward

        let mut layer_data = Vec::with_capacity(layers.len() - 1);
        let mut neuron_data = Vec::default();
        let weight_data = params.weights.clone();

        let mut neuron_count = 0;
        let mut weight_count = 0;
        for l in 1..layers.len() {
            let neurons_in_layer = layers[l].num_neurons();

            layer_data.push(LayerData { layer: layers[l], neuron_start_idx: neuron_count });

            let weights_per_neuron = layers[l - 1].num_neurons();
            for n in 0..neurons_in_layer {
                neuron_data.push(NeuronData { activation: None, bias: params.biases[n], weight_start_idx: weight_count });
                weight_count += weights_per_neuron;
            }

            neuron_count += neurons_in_layer;
        }

        NetworkData {
            layer_data,
            neuron_data,
            weight_data,
        }
    }

    #[inline]
    pub(super) fn output(&self) -> Vec<U> {
        let last_layer = self.layer_data.last().unwrap();
        let mut output = Vec::with_capacity(last_layer.layer.num_neurons());
        for n in 0..last_layer.layer.num_neurons() {
            output.push(self.neuron_data[n + last_layer.neuron_start_idx].activation.unwrap());
        }

        output
    }
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub(super) struct LayerData {
    pub(super) layer: Layer,
    pub(super) neuron_start_idx: usize,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub(super) struct NeuronData<U: RealMath> {
    pub(super) activation: Option<U>,
    pub(super) bias: U,
    pub(super) weight_start_idx: usize,
}