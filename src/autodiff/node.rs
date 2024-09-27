use super::grad_num::GradNum;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node<T: GradNum> {
    pub(super) partials: [T; 2],
    pub(super) parents: [usize; 2],
}