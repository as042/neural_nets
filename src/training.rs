use reverse::*;

use crate::{layer::*, network::Network, prelude::ActivationFn, running::RunSettings};

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct TrainingResults {
    grad: Vec<f64>,
    output: Vec<f64>,
    cost: f64,
}

impl TrainingResults {
    /// Returns the cost of the training data.
    #[inline]
    pub fn cost(&self) -> f64 {
        self.cost
    }
}

impl Network {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, settings: &RunSettings, desired_output: &Vec<f64>, eta: f64) -> TrainingResults {
        self.run(settings);

        let param_vec = self.create_param_vec();
        let data = self.create_data_vec(settings, desired_output);

        let tape = Tape::new();
        let params = tape.add_vars(&param_vec);

        let result = Self::diff_cost_eval(&params, &data);
        let full_gradient = result.grad();
        let grad = full_gradient.wrt(&params);

        self.adjust_params(&grad, eta, settings.clamp);

        TrainingResults {
            grad,
            output: self.output(),
            cost: self.total_cost(&desired_output),
        }
    }

    /// Creates the param vector that will be passed to the autodiff.
    #[inline]
    fn create_param_vec(&self) -> Vec<f64> {
        // this list transfers all parameters to the cost evaluator
        let mut params = Vec::<f64>::default();

        // add all actual nn params
        for w in self.weights() {
            params.push(w.value());
        }
        for n in self.neurons() {
            params.push(n.bias());
        }

        params
    }

    /// Creates the param vector that will be passed to the autodiff.
    #[inline]
    fn create_data_vec(&self, settings: &RunSettings, desired_output: &Vec<f64>) -> Vec<f64> {
        let input = &settings.input;
        
        // this list transfers all necessary paramters to the run function
        let mut data = Vec::<f64>::default();

        // add layer information
        for l in 0..self.num_layers() {
            data.push(self.nth_layer(l).num_neurons() as f64);
            if l > 0 { data.push(self.nth_layer(l).activation_fn().encode() as f64 )};
        }
        
        // a useless identifer to indicate where the layer info ends
        data.push(f64::NAN);

        // add the training data
        for i in input {
            data.push(i.clone());
        }
        for o in desired_output {
            data.push(o.clone());
        }

        // a useless identifer to indicate where the training data ends
        data.push(f64::NAN);

        // add necessary settings
        data.push(if settings.print { -1.0 } else { 1.0 } * if settings.clamp { 2.0 } else { 1.0 });

        data
    }

    #[inline]
    fn diff_cost_eval<'a>(params: &[Var<'a>], data: &[f64]) -> Var<'a> {
        // obtain layer info...
        let nan_idx = data.iter().position(|x| x.is_nan()).unwrap();
        let layer_info = &data[0..nan_idx];
        let mut data = data.to_vec();
        data.remove(nan_idx);

        // training data... 
        let nan_idx = data.iter().position(|x| x.is_nan()).unwrap();
        let training_data = &data[0..nan_idx];

        // misc
        let misc = data.last().unwrap();

        // create the differential network
        let mut net = DiffNetwork::new(params, layer_info);

        // create settings
        let mut input = Vec::default();
        for i in 0..net.layers[0].num_neurons() {
            input.push(training_data[i]);
        }
        let mut desired_output = Vec::default();
        for o in 0..net.layers.last().unwrap().num_neurons() {
            desired_output.push(training_data[o + net.layers[0].num_neurons()]);
        }

        let settings = RunSettings {
            input,
            clamp: misc.abs() == 2.0,
            print: misc.is_sign_negative(),
        };

        net.diff_run(settings);

        let total_cost = net.total_cost(&desired_output);

        if misc.is_sign_negative() {
            println!("Total cost: {}", total_cost);
        }

        total_cost
    }

    /// Computes total square error of `self`.
    #[inline]
    pub fn total_cost(&self, desired_output: &Vec<f64>) -> f64 {
        let mut total_cost = 0.0;
        let output = self.output();

        if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

        for j in 0..output.len() {
            total_cost += Self::cost(output[j], desired_output[j]);
        }

        total_cost
    }

    /// Computes square error.
    #[inline]
    fn cost(output: f64, desired_output: f64) -> f64 {
        (output - desired_output).powf(2.0)
    }

    // Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params(&mut self, grad: &Vec<f64>, eta: f64, clamp: bool) {
        let layers_len = self.layers().len();
        let weights_len = self.weights().len();
        for w in 0..weights_len {
            self.weights[w].value -= eta * grad[w + 2 + layers_len];
            if clamp { self.weights[w].value = self.weights[w].value.clamp(-1.0, 1.0); }
        }
        for b in 0..self.neurons.len() {
            self.neurons[b].bias -= eta * grad[b + weights_len + 2 + layers_len];
            if clamp { self.neurons[b].bias = self.neurons[b].bias.clamp(-1.0, 1.0); }
        }
    }
}



// here be dragons

#[derive(Clone, Debug, Default)]
pub(crate) struct DiffNetwork<'a> {
    pub(crate) layers: Vec<Layer>,
    pub(crate) neurons: Vec<DiffNeuron<'a>>,
    pub(crate) weights: Vec<DiffWeight<'a>>,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) struct DiffNeuron<'a> {
    pub(crate) activation: Var<'a>,
    pub(crate) bias: Var<'a>,
    pub(crate) num_weights: usize,
    pub(crate) weight_start_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) struct DiffWeight<'a> {
    pub(crate) value: Var<'a>,
}

impl<'a> DiffNetwork<'a> {
    /// Generates the differential network for backprop.
    #[inline]
    pub(crate) fn new(params: &[Var<'a>], layer_info: &[f64]) -> Self {
        // start creating network
        let mut net = DiffNetwork { layers: vec![], neurons: vec![], weights: vec![] };

        // construct layers
        net.layers.push(Layer { num_neurons: layer_info[0] as usize, ..Default::default() });
        let mut num_neurons = 0;
        for l in (1..layer_info.len()).step_by(2) {
            net.layers.push(Layer { 
                num_neurons: layer_info[l] as usize,
                neuron_start_idx: num_neurons,
                layer_type: LayerType::Comput,
                activation_fn: ActivationFn::decode(layer_info[l + 1])
            });

            for n in 0..num_neurons {
                net.neurons.push(DiffNeuron { activation: params[0], bias: params[b + num_weights], num_weights: , weight_start_idx: weights });
            }

            num_neurons += layer_info[l] as usize;
        }

        // count weights
        let mut num_weights = layer_info[0] as usize * layer_info[1] as usize;
        for l in (3..layer_info.len()).step_by(2) {
            num_weights += layer_info[l - 2] as usize * layer_info[l] as usize
        }

        // add weights and neurons
        for w in 0..num_weights {
            net.weights.push(DiffWeight { value: params[w] });
        }
        let mut weights = 0;
        for b in 0..num_neurons {
            
        }

        net
    }

    /// Returns the output of `self`.
    #[inline]
    pub fn output(&self) -> Vec<Var<'a>> {
        let mut vec = Vec::default();

        for n in 0..self.layers.last().unwrap().num_neurons() {
            vec.push(self.neurons[self.layers.last().unwrap().neuron_start_idx + n].activation)
        }

        vec
    }

    /// Computes total square error of `self`.
    #[inline]
    pub fn total_cost(&self, desired_output: &Vec<f64>) -> Var<'a> {
        let output = self.output();
        let mut total_cost = Self::cost(output[0], desired_output[0]);
        
        if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

        for j in 1..output.len() {
            total_cost = total_cost + Self::cost(output[j], desired_output[j]);
        }

        total_cost
    }

    /// Computes square error.
    #[inline]
    fn cost(output: Var<'a>, desired_output: f64) -> Var<'a> {
        (output - desired_output).powf(2.0)
    }
}

#[test]
fn simple_network_test() {
    let mut net = Network::new()
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

    assert_eq!(net.train(settings, &vec![0.59, 0.7], 0.1).cost(), 4.5606021083089745e-31);
}