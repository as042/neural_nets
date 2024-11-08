use crate::autodiff::real::Real;
use crate::network::params::Params;
use super::i64_to_real;

/// The data returned after training a `Network`.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TrainingResults<T: Real> {
    pub(super) params: Params<T>,
    pub(super) all_costs: Vec<Vec<Vec<T>>>,
    pub(super) avg_costs: Vec<Vec<T>>,
    pub(super) all_grads: Vec<Vec<Vec<T>>>,
}

impl<T: Real> TrainingResults<T> {
    /// Returns the new, optimized params.
    #[inline]
    pub fn params(&self) -> &Params<T> {
        &self.params
    }

    /// Returns all costs generated during training, organized by epoch then batch.
    #[inline]
    pub fn all_costs(&self) -> &Vec<Vec<Vec<T>>> {
        &self.all_costs
    }

    /// Returns the average cost of each batch generated during training, organized by epoch.
    #[inline]
    pub fn avg_costs(&self) -> &Vec<Vec<T>> {
        &self.avg_costs
    }

    /// Returns all grads generated during training, organized by epoch.
    #[inline]
    pub fn all_grads(&self) -> &Vec<Vec<Vec<T>>> {
        &self.all_grads
    }       

    /// Returns the average cost for each epoch.
    #[inline]
    pub fn epoch_cost(&self, dec_places: usize) -> Vec<T> {
        let two = T::one() + T::one();
        let ten = two * two * two + two;
        let mut ten_power = T::one();
        for _ in 0..dec_places {
            ten_power = ten_power * ten;
        }
        self.avg_costs
            .iter()
            .map(|x| x.iter().fold(T::zero(), |acc, &y| acc + y) / i64_to_real(x.len() as i64))
            .map(|x| (x * ten_power).round() / ten_power)
            .collect()
    }
}