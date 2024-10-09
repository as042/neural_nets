use super::real::Real;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node<T: Real> {
    pub(super) partials: [T; 2],
    pub(super) parents: [usize; 2],
}