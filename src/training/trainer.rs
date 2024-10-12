use crate::autodiff::{real::Real, var::Var};
use crate::network::{Network, params::Params};
use crate::training::{clamp_settings::ClampSettings, data_set::DataSet, training_settings::TrainingSettings};

use super::cost::CostFn;
use super::eta::Eta;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct NetworkTrainer<'t, T: Real> {
    network: Network,
    params: Option<Params<T>>,
    batch_size: Option<usize>,
    num_epochs: Option<usize>,
    cost_fn: CostFn<T, Var<'t, T>>,
    clamp_settings: ClampSettings<T>,
    eta: Eta<T>,
    data_set: Option<DataSet<T>>,
}

impl<'t, T: Real> NetworkTrainer<'t, T> {
    #[inline]
    pub fn new(network: Network) -> Self {
        NetworkTrainer {
            network,
            params: None,
            batch_size: None,
            num_epochs: None,
            cost_fn: CostFn::MSE,
            clamp_settings: ClampSettings {
                weight_min: T::MIN,
                weight_max: T::MAX,
                bias_min: T::MIN,
                bias_max: T::MAX,
            },
            eta: Eta::point_one(),
            data_set: None,
        }
    }

    #[inline]
    pub fn params(mut self, params: Params<T>) -> Self {
        self.params = Some(params);
        self
    }

    #[inline]
    pub fn training_settings(mut self, settings: TrainingSettings<'t, T>) -> Self {
        self.batch_size = Some(settings.batch_size);
        self.num_epochs = Some(settings.num_epochs);
        self.cost_fn = settings.cost_fn;
        self.clamp_settings = settings.clamp_settings;
        self.eta = settings.eta;
        self.data_set = Some(settings.data_set);
        self
    }

    #[inline]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    #[inline]
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = Some(n);
        self
    }

    #[inline]
    pub fn cost_fn(mut self, cost_fn: CostFn<T, Var<'t, T>>) -> Self {
        self.cost_fn = cost_fn;
        self
    }

    #[inline]
    pub fn clamp_settings(mut self, settings: ClampSettings<T>) -> Self {
        self.clamp_settings = settings;
        self
    }

    #[inline]
    pub fn weight_min(mut self, min: T) -> Self {
        self.clamp_settings.weight_min = min;
        self
    }

    #[inline]
    pub fn weight_max(mut self, max: T) -> Self {
        self.clamp_settings.weight_max = max;
        self
    }

    #[inline]
    pub fn bias_min(mut self, min: T) -> Self {
        self.clamp_settings.bias_min = min;
        self
    }

    #[inline]
    pub fn bias_max(mut self, max: T) -> Self {
        self.clamp_settings.bias_max = max;
        self
    }

    #[inline]
    pub fn eta(mut self, eta: Eta<T>) -> Self {
        self.eta = eta;
        self
    }

    #[inline]
    pub fn data_set(mut self, data_set: DataSet<T>) -> Self {
        self.data_set = Some(data_set);
        self
    }

    #[inline]
    pub fn train(self) -> Params<T> {
        if self.params.is_none() { panic!("Params must be explicitly set") };
        if self.batch_size.is_none() { panic!("Batch size must be explicitly set") };
        if self.num_epochs.is_none() { panic!("Num epochs must be explicitly set") };
        if self.data_set.is_none() { panic!("Data set must be explicitly set") };

        let settings = TrainingSettings {
            batch_size: self.batch_size.unwrap(),
            num_epochs: self.num_epochs.unwrap(),
            cost_fn: self.cost_fn,
            clamp_settings: self.clamp_settings,
            eta: self.eta,
            data_set: self.data_set.unwrap(),
        };

        self.network.train::<T, Var<T>>(&settings, self.params.unwrap())
    }
}

#[test]
fn test_trainer() {
    use crate::prelude::*;

    let net = Network::builder()
    .input_layer(5)
    .feed_forward_layer(ActivationFn::ReLU, 3)
    .feed_forward_layer(ActivationFn::Linear, 4)
    .build();

    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let desired_output = vec![0.1, -0.2, 0.3, -0.5];
    let data_set = DataSet::builder()
        .sample(input.clone(), desired_output.clone())
        .sample(vec![0.9, 0.12, 0.33, 0.48, 0.55], vec![-1.1, -2.2, 0.4, -0.21])
        .sample(vec![0.54, -1.2, -0.31, 0.41, 0.53], vec![1.6, -0.5, 0.12, -0.9])
        .build();

    let params = net.random_params::<f64>(Seed::Input(1.0));

    let res = net.run(&input, &params);
    println!("res1: {:?}", res);

    let optimized = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(1000)
        .cost_fn(CostFn::MSE)
        .weight_min(f64::MIN)
        .weight_max(f64::MAX)
        .bias_min(f64::MIN)
        .bias_max(f64::MAX)
        .eta(Eta::point_one())
        .train();

    let res = net.run(&input, &optimized);
    println!("res2: {:?}", res);
    println!("new params: {:?}", optimized);
}