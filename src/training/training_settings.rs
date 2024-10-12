use crate::autodiff::real::Real;
use crate::autodiff::var::Var;

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
    pub fn num_batches(&self) -> usize {
        (self.data_set.len() as f32 / self.batch_size as f32).ceil() as usize
    }
}