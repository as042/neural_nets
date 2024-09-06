use std::{cell::RefCell, f64::consts::E, ops::{Add, Div, Mul, Neg, Sub}};

#[derive(Clone, Debug)]
pub struct Grad {
    partials: Vec<f64>,
    num_inputs: usize,
}

impl Grad {
    #[inline]
    pub fn wrt(&self, var: VarP) -> f64 {
        self.partials[var.index]
    }

    #[inline]
    pub fn full(&self) -> &Vec<f64> {
        &self.partials
    }

    #[inline]
    pub fn inputs(&self) -> &[f64] {
        &self.partials[0..self.num_inputs]
    }
}

#[derive(Clone, Copy)]
struct Node {
    partials: [f64; 2],
    parents: [usize; 2],
}

#[derive(Clone, Default)]
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    num_inputs: RefCell<usize>,
}

#[derive(Clone, Copy)]
pub struct VarP<'t> {
    tape: &'t Tape,
    index: usize,
    val: f64,
}

impl Tape {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

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

        *self.num_inputs.borrow_mut() += 1;
        
        VarP {
            tape: self,
            index: len,
            val: value,
        }
    }

    #[inline]
    pub fn unary_op(&self, partial: f64, index: usize, new_value: f64) -> VarP {
        self.binary_op(partial, 0.0, index, index, new_value)
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

impl<'t> VarP<'t> {
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

        Grad { partials: grad, num_inputs: *self.tape.num_inputs.borrow() }
    }

    #[inline]
    pub fn recip(self) -> VarP<'t> {
        self.tape.unary_op(-self.val.powf(-2.0), self.index, self.val.recip())
    }

    #[inline]
    pub fn sin(self) -> VarP<'t> {
        self.tape.unary_op(self.val.cos(), self.index, self.val.sin())
    }

    #[inline]
    pub fn cos(self) -> VarP<'t> {
        self.tape.unary_op(-self.val.sin(), self.index, self.val.cos())
    }

    #[inline]
    pub fn tan(self) -> VarP<'t> {
        self.tape.unary_op(self.val.cos().recip().powf(2.0), self.index, self.val.tan())
    }

    #[inline]
    pub fn exp(self) -> VarP<'t> {
        self.tape.unary_op(self.val.exp(), self.index, self.val.exp())
    }

    #[inline]
    pub fn ln(self) -> VarP<'t> {
        self.log(E)
    }

    #[inline]
    pub fn log2(self) -> VarP<'t> {
        self.log(2.0)
    }

    #[inline]
    pub fn log10(self) -> VarP<'t> {
        self.log(10.0)
    }
}

// addition
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
        self.tape.unary_op(1.0, self.index, self.val + rhs)
    }
}

impl<'t> Add<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn add(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs + self
    }
}

// subtraction
impl<'t> Sub for VarP<'t> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: VarP<'t>) -> Self::Output {
        self.tape.binary_op(1.0, -1.0, self.index, rhs.index, self.val - rhs.val)
    }
}

impl<'t> Sub<f64> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self.tape.unary_op(1.0, self.index, self.val - rhs)
    }
}

impl<'t> Sub<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn sub(self, rhs: VarP<'t>) -> VarP<'t> {
        -rhs + self
    }
}

// multiplication
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
        self.tape.unary_op(rhs, self.index, self.val * rhs)
    }
}

impl<'t> Mul<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn mul(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs * self
    }
}

// division
impl<'t> Div for VarP<'t> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.val.recip(), self.val * -rhs.val.powf(-2.0), self.index, rhs.index, self.val / rhs.val)
    }
}

impl<'t> Div<f64> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self.tape.unary_op(rhs.recip(), self.index, self.val / rhs)
    }
}

impl<'t> Div<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn div(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs.tape.unary_op(self * -rhs.val.powf(-2.0), rhs.index, self / rhs.val)
    }
}

// negation
impl<'a> Neg for VarP<'a> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

// powers
pub trait Powf<T> {
    type Output;
    /// Calculate `powf` for self, where `other` is the power to raise `self` to.
    fn powf(self, other: T) -> Self::Output;
}

impl<'t> Powf<VarP<'t>> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn powf(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.val * self.val.powf(rhs.val - 1.0), self.val.powf(rhs.val) * self.val.ln(), self.index, rhs.index, self.val.powf(rhs.val))
    }
}

impl<'t> Powf<f64> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn powf(self, rhs: f64) -> Self::Output {
        self.tape.unary_op(rhs * self.val.powf(rhs - 1.0), self.index, self.val.powf(rhs))
    }
}

impl<'t> Powf<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn powf(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs.tape.unary_op(self.powf(rhs.val) * self.ln(), rhs.index, self.powf(rhs.val))
    }
}

// logs
pub trait Log<T> {
    type Output;
    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(self, other: T) -> Self::Output;
}

impl<'t> Log<VarP<'t>> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn log(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(
            (self.val * rhs.val.ln()).recip(),
            -self.val.ln() / (rhs.val * rhs.val.ln().powf(2.0)),
            self.index, rhs.index, self.val.log(rhs.val))
    }
}

impl<'t> Log<f64> for VarP<'t> {
    type Output = Self;

    #[inline]
    fn log(self, rhs: f64) -> Self::Output {
        self.tape.unary_op((self.val * rhs.ln()).recip(), self.index, self.val.log(rhs))
    }
}

impl<'t> Log<VarP<'t>> for f64 {
    type Output = VarP<'t>;

    #[inline]
    fn log(self, rhs: VarP<'t>) -> VarP<'t> {
        rhs.tape.unary_op((self * rhs.val.ln()).recip(), rhs.index, self.log(rhs.val))
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
    let tape = Tape::default();
    let x = tape.new_var(1.0);
    let y = tape.new_var(1.0);

    // z = -2x + xxxy + 2y
    let z = -2.0 * x + x * x * x * y + 2.0 * y;
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
    let z = -x + (x - 0.5) * y + y / x - 2.0 * x / (5.0 * y - 1.0) - 3.0 * y + 7.0;
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
    let z = x.powf(y) - y.powf(x) + (2.0 * x).powf(-y);
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
    let z = (2.0 * x * y).sin() * y.cos().sin() * y.tan() + x.tan().recip();
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
    let z = ((a.recip() / b.recip() + 50.0).log(c)).powf(x) + y * (a + x).sin() + 0.3 * b * (x + y + a + b + c).tan().powf(2.0) / (c + b.powf(y)).log2();
    let grad = z.backprop();

    assert_eq!(grad.inputs().iter().map(|x| (x * 1E5).round() / 1E5).collect::<Vec<f64>>(), 
        [12.46499, -0.16416, 7.83974, 0.31315, -1.12997]);
    assert_eq!(grad.full().len(), 28);
}