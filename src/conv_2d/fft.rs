use std::fmt::Display;

use ndarray::{prelude::*, Data, DataOwned};
use num::{traits::AsPrimitive, traits::NumAssign, Float, ToPrimitive};
use rustfft::num_traits::FromPrimitive;

use super::{c2cfft, fft_2d};

pub trait Conv2DFftExt<T: rustfft::FftNum + Float, S: Data> {
    fn conv_2d_fft(&self, kernel: &ArrayBase<S, Ix2>) -> Option<Array2<T>>;
}

impl<S, T> Conv2DFftExt<T, S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = T>,
    T: rustfft::FftNum + Float,
{
    fn conv_2d_fft(&self, kernel: &ArrayBase<S, Ix2>) -> Option<Array2<T>> {
        let (shape, mut ret) = conv_2d_fft_inner(self, kernel);
        // Some(ret.slice_mut(s!(..shape.0; 2, ..shape.1; 2)).to_owned())
        Some(ret.slice_mut(s!(..shape.0, ..shape.1)).to_owned())
    }
}

fn conv_2d_fft_inner<S, T>(
    data: &ArrayBase<S, Ix2>,
    kernel: &ArrayBase<S, Ix2>,
) -> ((usize, usize), Array2<T>)
where
    S: Data<Elem = T>,
    T: rustfft::FftNum + Float,
{
    let output_shape = (
        data.shape()[0] + kernel.shape()[0] - 1,
        data.shape()[1] + kernel.shape()[1] - 1,
    );

    let fft_shape = (good_size_c(output_shape.0), good_size_r(output_shape.1));

    let mut data_ext = Array2::zeros(fft_shape);
    data_ext
        .slice_mut(s!(..data.shape()[0], ..data.shape()[1]))
        .assign(data);

    let mut rp = realfft::RealFftPlanner::new();
    let mut cp = rustfft::FftPlanner::new();

    let data_spec = fft_2d::forward(&mut data_ext, &mut rp, &mut cp);

    let mut kernel_ext = Array2::zeros(fft_shape);
    kernel_ext
        .slice_mut(s![..kernel.shape()[0], ..kernel.shape()[1]])
        .assign(kernel);
    let kernel_spec = fft_2d::forward(&mut kernel_ext, &mut rp, &mut cp);

    let mut mul_spec = data_spec * kernel_spec;

    (
        output_shape,
        fft_2d::inverse(&mut mul_spec, &mut rp, &mut cp),
    )
}

pub fn good_size_cc(n: usize) -> usize {
    let mut best_fac = n.next_power_of_two();

    loop {
        let new_fac = best_fac / 4 * 3;
        match new_fac.cmp(&n) {
            std::cmp::Ordering::Less => break,
            std::cmp::Ordering::Equal => return n,
            std::cmp::Ordering::Greater => {
                best_fac = new_fac;
            }
        }
    }

    best_fac
}

pub fn good_size_rr(n: usize) -> usize {
    let n = n / 2;

    good_size_cc(n) * 2
}

pub fn good_size_c(n: usize) -> usize {
    if n <= 12 {
        return n;
    } else if n < 3000 {
        return good_size_cc(n);
    }

    let mut best_fac = 2 * n;
    let mut f11 = 1;
    while f11 < best_fac {
        let mut f117 = f11;
        while f117 < best_fac {
            let mut f1175 = f117;
            while f1175 < best_fac {
                let mut x = f1175;
                while x < n {
                    x *= 2;
                }
                loop {
                    match x.cmp(&n) {
                        std::cmp::Ordering::Less => x *= 3,
                        std::cmp::Ordering::Equal => return n,
                        std::cmp::Ordering::Greater => {
                            if x < best_fac {
                                best_fac = x;
                            }
                            if num::Integer::is_odd(&x) {
                                break;
                            }
                            x >>= 1;
                        }
                    }
                }
                f1175 *= 5;
            }
            f117 *= 7;
        }

        f11 *= 11;
    }

    best_fac
}

pub fn good_size_r(n: usize) -> usize {
    if n <= 6 {
        return n;
    } else if n < 3000 {
        return good_size_rr(n);
    }

    let mut best_fac = 2 * n;
    let mut f5 = 1;

    while f5 < best_fac {
        let mut x = f5;
        while x < n {
            x *= 2;
        }
        loop {
            match x.cmp(&n) {
                std::cmp::Ordering::Less => x *= 3,
                std::cmp::Ordering::Equal => return n,
                std::cmp::Ordering::Greater => {
                    if x < best_fac {
                        best_fac = x;
                    }
                    if num::Integer::is_odd(&x) {
                        break;
                    }
                    x >>= 1;
                }
            }
        }
        f5 *= 5;
    }

    best_fac
}

pub fn conv_2d_c2c<T, S: ndarray::Data<Elem = T>, SM: ndarray::Data<Elem = T>>(
    data: &ArrayBase<S, Ix2>,
    kernel: &ArrayBase<S, Ix2>,
) -> Option<Array2<T>>
where
    T: Copy
        + Clone
        + NumAssign
        + std::fmt::Debug
        + Display
        + Send
        + Sync
        + FromPrimitive
        + ToPrimitive
        + AsPrimitive<f32>,

    f64: std::convert::From<T>,
{
    let output_shape = (
        data.shape()[0] + kernel.shape()[0] - 1,
        data.shape()[1] + kernel.shape()[1] - 1,
    );

    let fft_shape = (good_size_c(output_shape.0), good_size_c(output_shape.1));

    let mut data_ext = Array2::zeros(fft_shape);
    data_ext
        .slice_mut(s!(..data.shape()[0], ..data.shape()[1]))
        .assign(data);

    // let mut rp = realfft::RealFftPlanner::new();
    // let mut cp = rustfft::FftPlanner::new();

    let mut planner = rustfft::FftPlanner::new();

    // let data_spec = fft_2d::forward(&data_ext, &mut rp, &mut cp);
    let data_spec = c2cfft::forward(&data_ext, &mut planner);

    let mut kernel_ext = Array2::zeros(fft_shape);
    kernel_ext
        .slice_mut(s![..kernel.shape()[0], ..kernel.shape()[1]])
        .assign(kernel);
    // let kernel_spec = fft_2d::forward(&kernel_ext, &mut rp, &mut cp);
    let kernel_spec = c2cfft::forward(&kernel_ext, &mut planner);

    let mut mul_spec = data_spec * kernel_spec;

    // Some(fft_2d::inverse(&mut mul_spec, &mut rp, &mut cp))
    Some(c2cfft::inverse(&mut mul_spec, &mut planner))
}

#[cfg(test)]
mod tests {
    use num::Integer;

    use super::*;

    #[test]
    fn basic_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let mut ret = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let output_pixels = array![
            [2, 2, 3, 1, 1],
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [0, 2, 2, 1, 1],
        ];

        for a in input_pixels.axis_iter(Axis(0)) {
            dbg!(a);
        }

        // let kernel = array![[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]];
        let mut kernel = array![[1, 0, 1], [0, 0, 1], [0, 1, 0]];
        kernel.as_slice_mut().unwrap().reverse();

        dbg!(input_pixels
            .mapv(|x| x as f64)
            .conv_2d_fft(&kernel.mapv(|x| x as f64))
            .unwrap()
            .mapv(|x| x.round() as i32));
        // dbg!(&conv_2d_c2c::<
        //     f64,
        //     ndarray::OwnedRepr<f64>,
        //     ndarray::OwnedRepr<f64>,
        // >(&input_pixels.mapv(|x| x as f64), &kernel.mapv(|x| x as f64)));
    }

    #[test]
    fn test_best_fft_len() {
        // let mut n = 93059;
        let mut n = 5020;
        dbg!(good_size_c(n));
        dbg!(good_size_cc(n));
        dbg!(good_size_r(n));
        dbg!(good_size_rr(n));
    }
}
