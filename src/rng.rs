use std::time::SystemTime;

use crate::autodiff::real::Real;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Seed<T> {
    #[default]
    OS,
    Input(T),
}

#[inline]
pub fn lehmer_rng<T: Real>(state: T) -> T {
    let one = T::one();
    let two = T::one() + T::one();
    let three = two + one;
    let four = three + one;
    let sixteen = four * four;

    // 4^7 * 3 - (4^5) + (4 * 4 * 3 * 3) - 1
    let num48271 = sixteen * sixteen * sixteen * four * three - sixteen * sixteen * four + sixteen * three * three - one;
    let num0x7fffffff = sixteen * sixteen * sixteen * sixteen * sixteen * sixteen * sixteen * four * two - one;
    
    (num48271 * state) % num0x7fffffff
}

#[inline]
pub fn os_seed<T: Real>() -> T {
    let system_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();

    let two = T::one() + T::one();
    let ten = two * two * two + two;
    let mut val = T::zero();
    for d in system_time.to_string().chars().map(|d| d.to_digit(10).unwrap()).rev().enumerate() {
        let mut ten_power = T::one();
        for _ in 0..d.0 {
            ten_power = ten_power * ten;
        }
        val = val + ten_power * match d.1 {
            0 => T::zero(),
            1 => T::one(),
            2 => two,
            3 => two + T::one(),
            4 => two + two,
            5 => two + two + T::one(),
            6 => ten - two - two,
            7 => ten - two - T::one(),
            8 => ten - two,
            9 => ten - T::one(),
            _ => panic!("Invalid digit"),
        };
    }

    val
}

#[inline]
pub fn shuffle<T: Clone>(vec: &mut Vec<T>, seed: Seed<f64>) {
    let mut rand = 0.0;
    if seed == Seed::OS {
        rand = os_seed();
    }
    else if let Seed::Input(val) = seed {
        if val <= f64::EPSILON { panic!("Seed must be greater than 0") };

        rand = val;
    }

    for i in 0..(vec.len() - 1) {
        let j = (rand as usize) % (vec.len() - i) + i;
        vec.swap(i, j);
    }
}