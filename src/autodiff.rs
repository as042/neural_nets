use std::{cell::RefCell, ops::{Add, Mul, Neg}};

#[derive(Clone)]
pub struct Grad {
    derivatives: Vec<f64>,
}

impl Grad {
    #[inline]
    pub fn wrt(&self, var: VarP) -> f64 {
        self.derivatives[var.index]
    }
}

#[derive(Clone, Copy)]
pub struct Node {
    pub partials: [f64; 2],
    pub parents: [usize; 2],
}

#[derive(Clone, Default)]
pub struct Tape {
    pub nodes: RefCell<Vec<Node>>,
}

#[derive(Clone, Copy)]
pub struct VarP<'t> {
    pub tape: &'t Tape,
    pub index: usize,
    pub val: f64,
}

impl Tape {
    #[inline]
    pub fn new_var(&self, value: f64) -> VarP {
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
            Node {
                partials: [0.0, 0.0],
                // for a single (input) variable, we point the parents to itself
                parents: [len, len],
            }
        );
        
        VarP {
            tape: self,
            index: len,
            val: value,
        }
    }

    #[inline]
    pub fn binary_op(&self, lhs_partial: f64, rhs_partial: f64, lhs_index: usize, rhs_index: usize, new_value: f64) -> VarP {
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
            Node {
                partials: [lhs_partial, rhs_partial],
                // for a single (input) variable, we point the parents to itself
                parents: [lhs_index, rhs_index],
            }
        );

        VarP {
            tape: self,
            index: len,
            val: new_value,
        }
    }
}

impl VarP<'_> {
    /// Perform back propagation
    #[inline]
    pub fn backprop(&self) -> Grad {
        // vector storing the gradients
        let tape_len = self.tape.nodes.borrow().len();
        let mut grad = vec![0.0; tape_len];
        grad[self.index] = 1.0;

        for i in (0..tape_len).rev() {
            let node = self.tape.nodes.borrow()[i];
            // increment gradient contribution to the left parent
            let lhs_dep = node.parents[0];
            let lhs_partial = node.partials[0];
            grad[lhs_dep] += lhs_partial * grad[i];

            // increment gradient contribution to the right parent
            // note that in cases of unary operations, because
            // partial was set to zero, it won't affect the computation
            let rhs_dep = node.parents[1];
            let rhs_partial = node.partials[1];
            grad[rhs_dep] += rhs_partial * grad[i];
        }

        Grad { derivatives: grad }
    }
}

impl<'t> Add for VarP<'t> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: VarP<'t>) -> Self::Output {
        self.tape.binary_op(1.0, 1.0, self.index, rhs.index, self.val + rhs.val)
    }
}

impl<'t> Add<f64> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        self.tape.binary_op(1.0, 0.0, self.index, self.index, self.val + rhs)
    }
}

impl<'t> Add<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn add(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs + self
    }
}

impl<'t> Mul for VarP<'t> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.val, self.val, self.index, rhs.index, self.val * rhs.val)
    }
}

impl<'t> Mul<f64> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self.tape.binary_op(rhs, 0.0, self.index, self.index, self.val * rhs)
    }
}

impl<'t> Mul<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn mul(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs * self
    }
}

impl<'a> Neg for VarP<'a> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self * -1.
    }
}

#[test]
fn test() {
    let tape = Tape::default();
    let x = tape.new_var(1.0);
    let y = tape.new_var(1.0);
    let z = -2.0 * x + x * x * x * y + 2.0 * y;
    let grad = z.backprop();

    println!("full grad: {:?}", grad.derivatives);
    println!("dz/dx of z = -2x + x^3 * y + 2y at x=1.0, y=1.0 is {}", grad.wrt(x));
    println!("dz/dy of z = -2x + x^3 * y + 2y at x=1.0, y=1.0 is {}", grad.wrt(y));
    panic!()
}