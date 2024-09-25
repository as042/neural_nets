use crate::prelude::{GradNum, Powf, Tape, Var};

use crate::{layer::*, network::Network, prelude::ActivationFn, running::RunSettings};

/// The data returned after training a `Network`.
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct TrainingResults<'t, T: GradNum> {
    grad: Vec<T>,
    output: Vec<Var<'t, T>>,
    cost: Var<'t, T>,
}

impl<'t, T: GradNum> TrainingResults<'t, T> {
    /// Returns the grad of the training data.
    #[inline]
    pub fn grad(&self) -> &Vec<T> {
        &self.grad
    }

    /// Returns the output of the training data.
    #[inline]
    pub fn output(&self) -> &Vec<Var<'t, T>> {
        &self.output
    }

    /// Returns the cost of the training data.
    #[inline]
    pub fn cost(&self) -> Var<'t, T> {
        self.cost
    }
}

impl<'t, T: GradNum> Network<'t, T> {    
    /// Runs `self` with the given input and adjusts params to minimize cost.
    #[inline]
    pub fn train(&mut self, settings: &RunSettings<'t, T>, cost_fn: fn(&Vec<Var<'t, T>>, &Vec<T>) -> Var<'t, T>, desired_output: &Vec<T>, eta: T) -> TrainingResults<T> {
        let tape = Tape::<T>::new();
        let weights: Vec<T> = self.weights().iter().map(|x| x.value().val()).collect();
        let biases: Vec<T> = self.neurons().iter().map(|x| x.bias().val()).collect();
        let _ = tape.new_vars(&[weights, biases].concat());
    
        self.run(settings);
        let output = self.output();
        let cost = cost_fn(&output, desired_output);

        let full_gradient = cost.backprop();
        let grad = full_gradient.wrt_inputs();

        self.adjust_params(grad, eta, settings.clamp);

        TrainingResults {
            grad: grad.to_vec(),
            output: self.output(),
            cost,
        }
    }

    // Adjusts weights and biases according to grad.
    #[inline]
    fn adjust_params(&mut self, grad: &[T], eta: T, clamp: bool) {
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

    // /// Returns the output of `self` but differential.
    // #[inline]
    // pub fn output(&self) -> Vec<Var<'a, T>> {
    //     let mut vec = Vec::default();

    //     for n in 0..self.layers.last().unwrap().num_neurons() {
    //         vec.push(self.neurons[self.layers.last().unwrap().neuron_start_idx + n].activation)
    //     }

    //     vec
    // }

    // /// Computes total square error of `self` but differential.
    // #[inline]
    // pub fn total_cost(&self, desired_output: &Vec<T>) -> Var<'a, T> {
    //     let output = self.output();
    //     let mut total_cost = Self::cost(output[0], desired_output[0]);
        
    //     if output.len() != desired_output.len() { panic!("Output layer must have same len as desired output") }

    //     for j in 1..output.len() {
    //         total_cost = total_cost + Self::cost(output[j], desired_output[j]);
    //     }

    //     total_cost
    // }

    // /// Computes square error but differential.
    // #[inline]
    // fn cost(output: Var<'a, T>, desired_output: T) -> Var<'a, T> {
    //     (output - desired_output).powf(T::one() + T::one())
    // }