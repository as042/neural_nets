use crate::autodiff::{grad_num::GradNum, var::Var};

use super::{layer::{Layer, LayerType}, params::Params};

#[derive(Clone, Default, Debug, PartialEq, PartialOrd)]
pub(super) struct NetworkData<'t, T: GradNum> {
    pub(super) layer_data: Vec<LayerData>,
    pub(super) neuron_data: Vec<NeuronData<'t, T>>,
    pub(super) weight_data: Vec<Var<'t, T>>,
}

impl<'t, T: GradNum> NetworkData<'t, T> {
    #[inline]
    pub(super) fn new(layers: &Vec<Layer>, params: &Params<'t, T>) -> Self {
        assert_eq!(layers[0].layer_type(), LayerType::Input); // first layer is input
        assert!(layers.len() > 1); // more than one layer
        assert!(!layers[1..].iter().any(|&x| x.layer_type != LayerType::FeedForward)); // all but the first layer are feed forward

        let mut layer_data = Vec::with_capacity(layers.len() - 1);
        let mut neuron_data = Vec::default();
        let weight_data = params.weights().to_vec();

        let mut neuron_count = 0;
        let mut weight_count = 0;
        for l in 1..layers.len() {
            let neurons_in_layer = layers[l].num_neurons();

            layer_data.push(LayerData { layer: layers[l], neuron_start_idx: neuron_count });

            let weights_per_neuron = layers[l - 1].num_neurons();
            for n in 0..neurons_in_layer {
                neuron_data.push(NeuronData { activation: None, bias: params.biases()[n], weight_start_idx: weight_count });
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
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub(super) struct LayerData {
    pub(super) layer: Layer,
    pub(super) neuron_start_idx: usize,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub(super) struct NeuronData<'t, T: GradNum> {
    pub(super) activation: Option<Var<'t, T>>,
    pub(super) bias: Var<'t, T>,
    pub(super) weight_start_idx: usize,
}