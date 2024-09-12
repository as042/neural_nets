use crate::prelude::{GradNum, Powf, Tape, Var};

use crate::{layer::*, network::Network, prelude::ActivationFn, running::RunSettings};

/// The data returned after training a `Network`.
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct TrainingResults<T: GradNum> {
    grad: Vec<T>,
    output: Vec<T>,
    cost: T,
}

impl<T: GradNum> TrainingResults<T> {
    /// Returns the grad of the training data.
    #[inline]
    pub fn grad(&self) -> &Vec<T> {
        &self.grad
    }

    /// Returns the output of the training data.
    #[inline]
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }

    /// Returns the cost of the training data.
    #[inline]
    pub fn cost(&self) -> T {
        self.cost
    }
}

impl<T: GradNum> Network<T> {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, settings: &RunSettings, desired_output: &Vec<T>, eta: T) -> TrainingResults<T> {
        self.run(settings);

        let weights_and_biases = self.weights_and_biases();

        let tape = Tape::new();
        let params = tape.new_vars(&[weights_and_biases.0, weights_and_biases.1].concat());

        let var_network = DiffNetwork::new(weights_and_biases.0, weights_and_biases.1, layer_info);

        // let result = 
        let full_gradient = result.grad();
        let grad = full_gradient.wrt(&params);

        self.adjust_params(&grad, eta, settings.clamp);

        TrainingResults {
            grad,
            output: self.output(),
            cost: self.total_cost(&desired_output),
        }
    }

    // Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params(&mut self, grad: &Vec<T>, eta: T, clamp: bool) {
        let weights_len = self.weights().len();
        for w in 0..weights_len {
            self.weights[w].value = self.weights[w].value - eta * grad[w];
            // if clamp { self.weights[w].value = self.weights[w].value.clamp(-1.0, 1.0); }
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias = self.neurons[b].bias - eta * grad[b + weights_len];
            // if clamp { self.neurons[b].bias = self.neurons[b].bias.clamp(-1.0, 1.0); }
        }
    }
}

/// Like a normal `Network` but differential.
#[derive(Clone)]
pub(crate) struct DiffNetwork<'a, T: GradNum> {
    pub(crate) layers: Vec<Layer>,
    pub(crate) neurons: Vec<DiffNeuron<'a, T>>,
    pub(crate) weights: Vec<DiffWeight<'a, T>>,
}

/// Like a normal `Neuron` but differential.
#[derive(Clone, Copy)]
pub(crate) struct DiffNeuron<'a, T: GradNum> {
    pub(crate) activation: Var<'a, T>,
    pub(crate) bias: Var<'a, T>,
    pub(crate) num_weights: usize,
    pub(crate) weight_start_idx: usize,
}

/// Like a normal `Weight` but differential.
#[derive(Clone, Copy)]
pub(crate) struct DiffWeight<'a, T: GradNum> {
    pub(crate) value: Var<'a, T>,
}

impl<'a, T: GradNum> DiffNetwork<'a, T> {
    /// Generates the differential network for backprop.
    #[inline]
    pub(crate) fn new(weights: &[Var<'a, T>], biases: &[Var<'a, T>], layer_info: &[f64]) -> Self {
        // start creating network
        let mut net = DiffNetwork { layers: vec![], neurons: vec![], weights: vec![] };

        // create the input layer
        net.layers = vec![Layer { num_neurons: layer_info[0] as usize, ..Default::default() }];

        let mut num_neurons = 0;
        let mut num_weights = 0;
        for l in (2..layer_info.len()).step_by(2) {
            let neurons_in_layer = layer_info[l] as usize;
            let layer_type = LayerType::Comput;
            let activation_fn = ActivationFn::decode(layer_info[l + 1]);
            net.layers.push(Layer { num_neurons: neurons_in_layer, neuron_start_idx: num_neurons, layer_type, activation_fn });

            let weights_per_neuron = layer_info[l - 2] as usize;
            for _ in 0..neurons_in_layer {
                net.neurons.push(DiffNeuron { 
                    activation: weights[0] * T::zero(), 
                    bias: weights[0], 
                    num_weights: weights_per_neuron, 
                    weight_start_idx: num_weights}
                );
                num_weights += weights_per_neuron;
            }

            num_neurons += neurons_in_layer;
        }

        // add weights and neurons
        for w in 0..num_weights {
            net.weights.push(DiffWeight { value: weights[w] });
        }
        for n in 0..num_neurons {
            net.neurons[n].bias = biases[num_weights];
        }

        net
    }

    /// Returns the output of `self` but differential.
    #[inline]
    pub fn output(&self) -> Vec<Var<'a, T>> {
        let mut vec = Vec::default();

        for n in 0..self.layers.last().unwrap().num_neurons() {
            vec.push(self.neurons[self.layers.last().unwrap().neuron_start_idx + n].activation)
        }

        vec
    }

    /// Computes total square error of `self` but differential.
    #[inline]
    pub fn total_cost(&self, desired_output: &Vec<T>) -> Var<'a, T> {
        let output = self.output();
        let mut total_cost = Self::cost(output[0], desired_output[0]);
        
        if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

        for j in 1..output.len() {
            total_cost = total_cost + Self::cost(output[j], desired_output[j]);
        }

        total_cost
    }

    /// Computes square error but differential.
    #[inline]
    fn cost(output: Var<'a, T>, desired_output: T) -> Var<'a, T> {
        (output - desired_output).powf(T::one() + T::one())
    }
}