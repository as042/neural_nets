pub mod activation_fn;
pub mod layer;
pub mod layout;
pub mod network_builder;
pub mod network_data;
pub mod params;
pub mod running;
pub mod run_results;

use std::fmt::Display;

use crate::autodiff::real::Real;
use crate::rng::Seed;
use crate::training::trainer::NetworkTrainer;

use layout::*;
use network_builder::NetworkBuilder;
use params::Params;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Network {
    layout: Layout,
}

impl Network {
    #[inline]
    pub fn new(layout: Layout) -> Self {
        Network { 
            layout, 
        }
    }

    #[inline]
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[inline]
    pub fn default_params<T: Real>(&self) -> Params<T> {
        Params::default_params(self.layout())
    }

    #[inline]
    pub fn random_params<T: Real>(&self, seed: Seed<T>) -> Params<T> {
        Params::random_params(self.layout(), seed)
    }

    #[inline]
    pub fn trainer<T: Real>(&self) -> NetworkTrainer<T> {
        NetworkTrainer::new(self.clone())
    }
}

impl Display for Network {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.layout)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn simple_network_test() {
        let layout = Layout::builder()
            .input_layer(5)
            .feed_forward_layer(ActivationFn::ReLU, 3)
            .feed_forward_layer(ActivationFn::Linear, 4)
            .build();

        let net = Network::new(layout);

        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let desired_output = vec![0.1, -0.2, 0.3, -0.5];
        let data_set = DataSet::new(vec![input.clone()], vec![desired_output.clone()]);
        let params = net.random_params::<f64>(Seed::OS);

        let res = net.run(&input, &params);
        println!("res1: {:?}", res);

        let train_res = net.train::<f64>(&TrainingSettings {
            batch_size: 1,
            num_epochs: 100,
            cost_fn: CostFn::MSE,
            clamp_settings: ClampSettings::new(-1.0, 1.0, -1.0, 1.0),
            eta: Eta::Const(0.1),
            data_set,
            stoch_shuffle_seed: Seed::Input(5.0),
        }, params);

        let res = net.run(&input, &train_res.params());
        println!("res2: {:?}", res);
    }
}