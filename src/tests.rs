use crate::{autodiff::real::Real, prelude::*};

pub trait RoundTo {
    fn round_to(self, dec_places: usize) -> Self;
}

impl<T: Real> RoundTo for T {
    #[inline]
    fn round_to(self, dec_places: usize) -> Self {
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        let mut ten_power = T::one();
        for _ in 0..dec_places {
            ten_power = ten_power * ten;
        }

        (self * ten_power).round() / ten_power
    }
}

pub fn identity_data_set() -> DataSet<f64> {
    let mut data_set = DataSet::builder();

    for i in 0..10000 {
        data_set = data_set.sample(vec![i as f64 / 100.0], vec![i as f64 / 100.0]);
    }

    data_set.build()
}

pub fn polynomial_arithmetic_data_set() -> DataSet<f64> {
    let mut data_set = DataSet::builder();

    for x in 0..10 {
        for a in 0..10 {
            for b in 0..10 {
                for c in 0..10 {
                    for d in 0..10 {
                        let x_val = x as f64 - 5.0;
                        let a_val = a as f64 - 5.0;
                        let b_val = b as f64 - 5.0;
                        let c_val = c as f64 - 5.0;
                        let d_val = d as f64 - 5.0;
                        let y = a_val*x_val.powf(3.0) + b_val*x_val.powf(2.0) + c_val*x_val + d_val;
                        data_set = data_set.sample(vec![x_val, a_val, b_val, c_val, d_val], vec![y]);
                    }
                }
            }
        }
    }

    data_set.build()
}

pub fn linear_identity_test() {
    let net = Network::builder()
        .input_layer(1)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = identity_data_set();

    let params = net.random_params::<f64>(Seed::OS);

    let train_res = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(1000)
        .cost_fn(CostFn::MSE)
        .clamp_settings(ClampSettings::NO_CLAMP)
        .eta(Eta::Const(1E-5))
        .stoch_shuffle_seed(Seed::OS)
        .train();

    println!("costs: {:?}", train_res.epoch_cost(5));
    println!("new params: {:?}", train_res.params());
}

pub fn identity_test() {
    let net = Network::builder()
        .input_layer(1)
        .feed_forward_layer(ActivationFn::Sigmoid, 10)
        .feed_forward_layer(ActivationFn::Sigmoid, 10)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = identity_data_set();

    let params = net.random_params::<f64>(Seed::OS);

    let train_res = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(1000)
        .cost_fn(CostFn::MSE)
        .clamp_settings(ClampSettings::NO_CLAMP)
        .eta(Eta::Const(1E-5))
        .stoch_shuffle_seed(Seed::OS)
        .train();

    println!("costs: {:?}", train_res.epoch_cost(5));
    println!("new params: {:?}", train_res.params());
}

pub fn cubic_calculator() {
    let net = Network::builder()
        .input_layer(5)
        .feed_forward_layer(ActivationFn::Sigmoid, 10)
        .feed_forward_layer(ActivationFn::Sigmoid, 10)
        .feed_forward_layer(ActivationFn::Linear, 1)
        .build();

    let data_set = polynomial_arithmetic_data_set();

    let params = net.random_params::<f64>(Seed::OS);

    let train_res = net.trainer()
        .data_set(data_set)
        .params(params)
        .batch_size(1)
        .num_epochs(1000)
        .cost_fn(CostFn::MSE)
        .clamp_settings(ClampSettings::NO_CLAMP)
        .eta(Eta::Const(1E-5))
        .stoch_shuffle_seed(Seed::OS)
        .train();

    println!("costs: {:?}", train_res.epoch_cost(5));
    println!("new params: {:?}", train_res.params());
}