use crate::autodiff::grad_num::GradNum;

use super::clamp_settings::ClampSettings;
use super::cost::CostFn;
use super::eta::Eta;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrainingSettings<'t, T: GradNum> {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub cost_fn: CostFn<'t, T>,
    pub clamp_settings: ClampSettings<T>,
    pub eta: Eta<T>,
    pub input_set: Vec<Vec<T>>,
    pub output_set: Vec<Vec<T>>,
}

impl<'t, T: GradNum> TrainingSettings<'t, T> {
    #[inline]
    pub fn batch_size(&self) -> &usize {
        &self.batch_size
    }

    #[inline]
    pub fn num_epochs(&self) -> &usize {
        &self.num_epochs
    }

    #[inline]
    pub fn cost_fn(&self) -> &CostFn<'t, T> {
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
    pub fn input_set(&self) -> &Vec<Vec<T>> {
        &self.input_set
    }

    #[inline]
    pub fn output_set(&self) -> &Vec<Vec<T>> {
        &self.output_set
    }
}