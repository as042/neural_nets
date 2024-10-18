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
pub fn shuffle<T: Real>(vec: &mut Vec<usize>, seed: Seed<T>) {
    let mut rand = T::zero();
    if seed == Seed::OS {
        rand = os_seed();
    }
    else if let Seed::Input(val) = seed {
        if val <= T::zero() { panic!("Seed must be greater than 0") };

        rand = val;
    }

    for i in 0..(vec.len() - 1) {
        let j = real_to_i64(rand) as usize % (vec.len() - i) + i;
        vec.swap(i, j);
    }
}

#[inline]
fn real_to_i64<T: Real>(mut x: T) -> i64 {
    let two = T::one() + T::one();
    let ten = two * two * two + two;    

    let sign = x.signum();
    x = x.abs();
    let mut val = 0;

    let mut i = T::zero();
    let mut j = 0;
    while i < x.log10().trunc() + T::one() {
        let digit = (x / ten.powf(i)).trunc() % ten; 

        val += 10i64.pow(j) * if digit == T::zero() { 0 }
            else if digit == T::one() { 1 }
            else if digit == two { 2 }
            else if digit == two + T::one() { 3 }
            else if digit == two + two { 4 }
            else if digit == two + two + T::one() { 5 }
            else if digit == ten - two - two { 6 }
            else if digit == ten - two - T::one() { 7 }
            else if digit == ten - two { 8 }
            else if digit == ten - T::one() { 9 }
            else { panic!("Invalid digit") };

        i = i + T::one();
        j += 1;
    } 

    val * if sign == T::one() { 1 } else { -1 }
}

#[test]
fn test_lehmer_rng() {
    let s = 123432.0;
    let s2 = lehmer_rng(s);
    let s3 = lehmer_rng(s2);

    assert_ne!(s, s2);
    assert_ne!(s2, s3);
    assert_ne!(s, s3);
}

#[test]
fn test_os_seed() {
    let s: f64 = os_seed();

    std::thread::sleep(std::time::Duration::from_micros(1));

    let s2 = os_seed();

    assert_ne!(s, s2);
}

#[test]
fn test_shuffle() {
    let v1 = vec![0, 1, 2, 3, 4];
    let mut v2 = v1.clone();
    shuffle(&mut v2, Seed::Input(1.0));

    assert_ne!(v1, v2);
    assert_eq!(v1.iter().fold(0, |acc, x| acc + x), v1.iter().fold(0, |acc, x| acc + x));
}

#[test]
fn test_real_to_i64() {
    assert_eq!(real_to_i64(69.0), 69);
    assert_eq!(real_to_i64(1000012428f64), 1000012428);
    assert_eq!(real_to_i64(10173480128374f64), 10173480128374);
    assert_eq!(real_to_i64(-128397.0), -128397);
    assert_eq!(real_to_i64(-0.0), 0);
    assert_eq!(real_to_i64(10.5), 10);
    assert_eq!(real_to_i64(199.9), 199);
}