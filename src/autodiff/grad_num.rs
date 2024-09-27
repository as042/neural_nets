// I would like to basically implement this myself to cutdown on dependencies
use num_traits::real::Real;

pub trait GradNum: Real + Default {}

impl<T> GradNum for T where T: Real + Default {}