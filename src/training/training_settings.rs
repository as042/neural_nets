use crate::autodiff::real::Real;
use crate::autodiff::var::Var;
use crate::rng::Seed;

use super::clamp_settings::ClampSettings;
use super::cost::CostFn;
use super::data_set::DataSet;
use super::eta::Eta;

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct TrainingSettings<'t, T: Real> {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub cost_fn: CostFn<T, Var<'t, T>>,
    pub clamp_settings: ClampSettings<T>,
    pub eta: Eta<T>,
    pub data_set: DataSet<T>,
    pub stoch_shuffle_seed: Seed<T>,
}

impl<'t, T: Real> TrainingSettings<'t, T> {
    #[inline]
    pub fn batch_size(&self) -> &usize {
        &self.batch_size
    }

    #[inline]
    pub fn num_epochs(&self) -> &usize {
        &self.num_epochs
    }

    #[inline]
    pub fn cost_fn(&self) -> &CostFn<T, Var<'t, T>> {
        &self.cost_fn
    }
    
    #[inline]
    pub fn clamp_settings(&self) -> &ClampSettings<T> {
        &self.clamp_settings
    }

    #[inline]
    pub fn eta(&self) -> &Eta<T>{
        &self.eta
    }

    #[inline]
    pub fn data_set(&self) -> &DataSet<T> {
        &self.data_set
    }

    #[inline]
    pub fn stoch_shuffle_seed(&self) -> &Seed<T>{
        &self.stoch_shuffle_seed
    }

    #[inline]
    pub fn num_batches(&self) -> usize {
        (self.data_set.len() as f32 / self.batch_size as f32).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_batches() {
        let data_set = DataSet::<f64>::builder()
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .sample(vec![], vec![])
            .build();

        let settings = TrainingSettings {
            batch_size: 4,
            num_epochs: 3,
            cost_fn: CostFn::MAE,
            clamp_settings: ClampSettings::default(),
            eta: Eta::Const(0.00001),
            data_set,
            stoch_shuffle_seed: Seed::OS
        };

        assert_eq!(settings.num_batches(), 3);
    }
}