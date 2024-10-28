use std::cell::RefCell;

use super::real::Real;
use super::node::Node;
use super::var::Var;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Tape<T: Real> {
    pub(super) nodes: RefCell<Vec<Node<T>>>,
    pub(super) num_inputs: RefCell<usize>,
}

impl<T: Real> Default for Tape<T> {
    #[inline]
    fn default() -> Self {
        Tape {
            nodes: vec![].into(),
            num_inputs: 0.into(),
        }
    }
}

impl<T: Real> Tape<T> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn new_var(&self, value: T) -> Var<T> {
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
            Node {
                partials: [T::zero(), T::zero()],
                // for a single (input) variable, we point the parents to itself
                parents: [len, len],
            }
        );

        *self.num_inputs.borrow_mut() += 1;
        
        Var {
            tape: self,
            index: len,
            val: value,
        }
    }

    #[inline]
    pub fn new_vars(&self, values: &Vec<T>) -> Vec<Var<T>> {
        let mut vec = Vec::default();
        for v in values {
            vec.push(self.new_var(*v));
        }

        vec
    }

    #[inline]
    pub fn unary_op(&self, partial: T, index: usize, new_value: T) -> Var<T> {
        self.binary_op(partial, T::zero(), index, index, new_value)
    }

    #[inline]
    pub fn binary_op(&self, lhs_partial: T, rhs_partial: T, lhs_index: usize, rhs_index: usize, new_value: T) -> Var<T> {
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
            Node {
                partials: [lhs_partial, rhs_partial],
                // for a single (input) variable, we point the parents to itself
                parents: [lhs_index, rhs_index],
            }
        );

        Var {
            tape: self,
            index: len,
            val: new_value,
        }
    }
}