use std::{cell::RefCell, ops::{Add, Div, Mul, Neg, Sub}};
use num_traits::real::Real;

pub trait GradNum: Real {}

impl<T> GradNum for T where T: Real {}

#[derive(Clone, Debug)]
pub struct Grad<T: GradNum> {
    partials: Vec<T>,
    num_inputs: usize,
}

impl<T: GradNum> Grad<T> {
    #[inline]
    pub fn wrt(&self, var: Var<T>) -> T {
        self.partials[var.index]
    }

    #[inline]
    pub fn full(&self) -> &Vec<T> {
        &self.partials
    }

    #[inline]
    pub fn inputs(&self) -> &[T] {
        &self.partials[0..self.num_inputs]
    }
}

#[derive(Clone, Copy)]
struct Node<T: GradNum> {
    partials: [T; 2],
    parents: [usize; 2],
}

#[derive(Clone)]
pub struct Tape<T: GradNum> {
    nodes: RefCell<Vec<Node<T>>>,
    num_inputs: RefCell<usize>,
}

impl<T: GradNum> Default for Tape<T> {
    fn default() -> Self {
        Tape {
            nodes: vec![].into(),
            num_inputs: 0.into(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Var<'t, T: GradNum> {
    tape: &'t Tape<T>,
    index: usize,
    val: T,
}

impl<T: GradNum> Tape<T> {
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
    pub fn unary_op(&self, partial: T, index: usize, new_value: T) -> VarP<T> {
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

impl<'t, T: GradNum> Var<'t, T> {
    /// Perform back propagation
    #[inline]
    pub fn backprop(&self) -> Grad<T> {
        // vector storing the gradients
        let tape_len = self.tape.nodes.borrow().len();
        let mut grad = vec![T::zero(); tape_len];
        grad[self.index] = T::one();

        for i in (0..tape_len).rev() {
            let node = self.tape.nodes.borrow()[i];
            // increment gradient contribution to the left parent
            let lhs_dep = node.parents[0];
            let lhs_partial = node.partials[0];
            let grad_i = grad[i];
            grad[lhs_dep] = grad[lhs_dep] + lhs_partial * grad_i;

            // increment gradient contribution to the right parent
            // note that in cases of unary operations, because
            // partial was set to zero, it won't affect the computation
            let rhs_dep = node.parents[1];
            let rhs_partial = node.partials[1];
            let grad_i = grad[i];
            grad[rhs_dep] = grad[rhs_dep] + rhs_partial * grad_i;
        }

        Grad { partials: grad, num_inputs: *self.tape.num_inputs.borrow() }
    }

    #[inline]
    pub fn recip(self) -> Var<'t, T> {
        self.tape.unary_op(-T::one() / (self.val * self.val), self.index, self.val.recip())
    }

    #[inline]
    pub fn sin(self) -> Var<'t, T> {
        self.tape.unary_op(self.val.cos(), self.index, self.val.sin())
    }

    #[inline]
    pub fn cos(self) -> Var<'t, T> {
        self.tape.unary_op(-self.val.sin(), self.index, self.val.cos())
    }

    #[inline]
    pub fn tan(self) -> Var<'t, T> {
        let cos_val = self.val.cos();
        self.tape.unary_op(T::one() / (cos_val * cos_val), self.index, self.val.tan())
    }

    #[inline]
    pub fn exp(self) -> Var<'t, T> {
        self.tape.unary_op(self.val.exp(), self.index, self.val.exp())
    }

    #[inline]
    pub fn ln(self) -> Var<'t, T> {
        let e = T::exp(T::one());
        self.log(e)
    }

    #[inline]
    pub fn log2(self) -> Var<'t, T> {
        let two = T::one() + T::one();
        self.log(two)
    }

    #[inline]
    pub fn log10(self) -> Var<'t, T> {
        // I know this seems absolutely insane, but I believe it is the 
        // way to do it with the fewest operations. ~MAR
        let two = T::one() + T::one();
        let eight = two * two * two;
        let ten = eight + two;
        self.log(ten)
    }
}

// addition
impl<'t, T: GradNum> Add for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Var<'t, T>) -> Self::Output {
        self.tape.binary_op(T::one(), T::one(), self.index, rhs.index, self.val + rhs.val)
    }
}

impl<'t, T: GradNum> Add<T> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.tape.unary_op(T::one(), self.index, self.val + rhs)
    }
}

// subtraction
impl<'t, T: GradNum> Sub for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Var<'t, T>) -> Self::Output {
        self.tape.binary_op(T::one(), -T::one(), self.index, rhs.index, self.val - rhs.val)
    }
}

impl<'t, T: GradNum> Sub<T> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.tape.unary_op(T::one(), self.index, self.val - rhs)
    }
}

// multiplication
impl<'t, T: GradNum> Mul for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.val, self.val, self.index, rhs.index, self.val * rhs.val)
    }
}

impl<'t, T: GradNum> Mul<T> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.tape.unary_op(rhs, self.index, self.val * rhs)
    }
}

// division
impl<'t, T: GradNum> Div for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.val.recip(), self.val * -T::one() / (rhs.val * rhs.val), self.index, rhs.index, self.val / rhs.val)
    }
}

impl<'t, T: GradNum> Div<T> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.tape.unary_op(rhs.recip(), self.index, self.val / rhs)
    }
}

// negation
impl<'a, T: GradNum> Neg for Var<'a, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self * -T::one()
    }
}

// powers
pub trait Powf<T> {
    type Output;
    /// Calculate `powf` for self, where `other` is the power to raise `self` to.
    fn powf(self, other: T) -> Self::Output;
}

impl<'t, T: GradNum> Powf<Var<'t, T>> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn powf(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.val * self.val.powf(rhs.val - T::one()), self.val.powf(rhs.val) * self.val.ln(), self.index, rhs.index, self.val.powf(rhs.val))
    }
}

impl<'t, T: GradNum> Powf<T> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn powf(self, rhs: T) -> Self::Output {
        self.tape.unary_op(rhs * self.val.powf(rhs - T::one()), self.index, self.val.powf(rhs))
    }
}

impl<'t, T: GradNum> Powf<Var<'t, T>> for T {
    type Output = Var<'t, T>;

    #[inline]
    fn powf(self, rhs: Var<'t, T>) -> Self::Output {
        rhs.tape.unary_op(self.powf(rhs.val) * self.ln(), rhs.index, self.powf(rhs.val))
    }
}

// logs
pub trait Log<T> {
    type Output;
    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(self, other: T) -> Self::Output;
}

impl<'t, T: GradNum> Log<Var<'t, T>> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn log(self, rhs: Self) -> Self::Output {
        let rhs_ln: T = rhs.val.ln();
        self.tape.binary_op(
            (self.val * rhs_ln).recip(),
            -self.val.ln() / (rhs.val * rhs_ln * rhs_ln),
            self.index, rhs.index, self.val.log(rhs.val))
    }
}

impl<'t, T: GradNum> Log<T> for Var<'t, T> {
    type Output = Self;

    #[inline]
    fn log(self, rhs: T) -> Self::Output {
        self.tape.unary_op((self.val * rhs.ln()).recip(), self.index, self.val.log(rhs))
    }
}

impl<'t, T: GradNum> Log<Var<'t, T>> for T {
    type Output = Var<'t, T>;

    #[inline]
    fn log(self, rhs: Var<'t, T>) -> Var<'t, T> {
        let rhs_ln: T = rhs.val.ln();
        rhs.tape.unary_op(-self.ln() / (rhs.val * rhs_ln * rhs_ln), rhs.index, self.log(rhs.val))
    }
}

#[test]
fn one_input_test() {
    let tape = Tape::new();
    let x = tape.new_var(-100.0);

    // z = xx + x
    let z = x * x + x;
    let grad = z.backprop();

    assert_eq!(grad.inputs(), [-199.0]);
}

#[test]
fn basic_test() {
    let tape: Tape<f64> = Tape::default();
    let x = tape.new_var(1.0);
    let y = tape.new_var(1.0);

    // -2x + xxxy + 2y
    let z = x * -2.0 + x * x * x * y + y * 2.0;
    let grad = z.backprop();

    println!("full grad: {:?}", grad.full());
    println!("grad: {:?}", grad.inputs());
    println!("dz/dx of z = -2x + x^3 * y + 2y at x=1.0, y=1.0 is {}", grad.wrt(x));
    println!("dz/dy of z = -2x + x^3 * y + 2y at x=1.0, y=1.0 is {}", grad.wrt(y));
    
    assert_eq!(grad.inputs(), [1.0, 3.0]);
}

#[test]
fn basic_arith_test() {
    let tape = Tape::new();
    let x = tape.new_var(2.0);
    let y = tape.new_var(3.0);

    // z = -x + y(x - 0.5) + y/x - 2x/(5y - 1) - 3y + 7
    let z = -x + (x - 0.5) * y + y / x - x * 2.0 / (y * 5.0 - 1.0) - y * 3.0 + 7.0;
    let grad = z.backprop();

    println!("full grad: {:?}", grad.full());

    assert_eq!(grad.inputs().iter().map(|x| (x * 1E5).round() / 1E5).collect::<Vec<f64>>(), [1.10714, -0.89796]);
}

#[test]
fn powf_test() {
    let tape = Tape::new();
    let x = tape.new_var(-1.0);
    let y = tape.new_var(2.0);

    // z = x^y - y^x + (2x)^-y
    let z = x.powf(y) - y.powf(x) + (x * 2.0).powf(-y);
    let grad = z.backprop();

    // grad wrt y is nan
    assert_eq!((grad.wrt(x) * 1E5).round() / 1E5, -1.84657);
}

#[test]
fn log_test() {
    let tape = Tape::new();
    let x = tape.new_var(3.0);
    let y = tape.new_var(5.0);

    // z = log_y(x) + y*log_2(x) + x*log_10(y) + xy*ln(x + y)
    let z = x.log(y) + y * x.log2() + x * y.log10() + x * y * (x + y).ln();
    let grad = z.backprop();

    assert_eq!(grad.inputs().iter().map(|x| (x * 1E5).round() / 1E5).collect::<Vec<f64>>(), [15.58278, 9.87404]);
}

#[test]
fn trig_test() {
    let tape = Tape::new();
    let x = tape.new_var(3.141);
    let y = tape.new_var(2.712);

    // z = sin(2xy)*sin(cos(y))*tan(y) + 1/tan(x)
    let z = (x * y * 2.0).sin() * y.cos().sin() * y.tan() + x.tan().recip();
    let grad = z.backprop();

    assert_eq!(grad.inputs().iter().map(|x| (x * 1E3).round() / 1E3).collect::<Vec<f64>>(), [-2847070.909, 0.269]);
}

#[test]
fn three_input_test() {
    let tape = Tape::new();
    let x = tape.new_var(1.5);
    let y = tape.new_var(3.0);
    let a = tape.new_var(4.0);

    // z = x^2 + ya + a 
    let z = x.powf(2.0) + y * a + a;
    let grad = z.backprop();

    assert_eq!(grad.inputs(), [3.0, 4.0, 4.0]);
}

#[test]
fn complex_test() {
    let tape = Tape::new();
    let x = tape.new_var(2.1);
    let y = tape.new_var(8.03);
    let a = tape.new_var(3.912);
    let b = tape.new_var(0.13);
    let c = tape.new_var(5.58);

    // z = log_c^x((1 / a) / (1 / b) + 50) + y*sin(a + x) + 0.3b*tan^2(x + y + a + b + c) / log_2(c + b^y)
    let z = ((a.recip() / b.recip() + 50.0).log(c)).powf(x) + y * (a + x).sin() + b * 0.3 * (x + y + a + b + c).tan().powf(2.0) / (c + b.powf(y)).log2();
    let grad = z.backprop();

    assert_eq!(grad.inputs().iter().map(|x| (x * 1E5).round() / 1E5).collect::<Vec<f64>>(), 
        [12.46499, -0.16416, 7.83974, 0.31315, -1.12997]);
    assert_eq!(grad.full().len(), 28);
}

#[test]
fn f32_test() {
    let tape = Tape::new();
    let x = tape.new_var(2f32);
    let y = tape.new_var(4f32);

    // log_x(1.5) + 2^y
    let z = Log::log(1.5, x) + Powf::powf(2.0, y);
    let grad = z.backprop();

    assert_eq!(grad.inputs().iter().map(|x| (x * 1E5).round() / 1E5).collect::<Vec<f32>>(), [-0.42196, 11.09036]);
}