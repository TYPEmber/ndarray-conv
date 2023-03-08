use std::fmt::Display;

use ndarray::{prelude::*, Data, DataMut, DataOwned};
use num::{traits::AsPrimitive, traits::NumAssign, Float, ToPrimitive};
use rustfft::num_traits::FromPrimitive;

use crate::{BorderType, ExplicitPadding, PaddingMode, PaddingSize};

use super::{conv_2d_inner, fft_2d, padding};

const N: usize = 2;

pub trait Conv2DFftExt<T: rustfft::FftNum + Float + NumAssign, S: Data, SK: Data> {
    fn conv_2d_fft(
        &self,
        kernel: &ArrayBase<SK, Ix2>,
        conv_type: PaddingSize<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array2<T>>;
}

impl<S, SK, T> Conv2DFftExt<T, S, SK> for ArrayBase<S, Ix2>
where
    S: Data<Elem = T>,
    SK: Data<Elem = T>,
    T: rustfft::FftNum + Float + NumAssign,
{
    fn conv_2d_fft(
        &self,
        kernel: &ArrayBase<SK, Ix2>,
        padding_size: PaddingSize<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array2<T>> {
        let data_size = [self.shape()[0], self.shape()[1]];
        let kernel_size = [kernel.shape()[0], kernel.shape()[1]];

        let kernel = Array2::from_shape_vec(
            kernel_size,
            kernel.as_slice().unwrap().iter().rev().copied().collect(),
        )
        .unwrap();

        let padding_size = padding_size.unfold(&kernel_size);

        let data_ext = pad_fft(self, &padding_size, &padding_mode)?;
        let mut kernel_ext = Array2::from_elem(data_ext.dim(), T::zero());
        kernel_ext
            .slice_mut(s![..kernel_size[0], ..kernel_size[1]])
            .assign(&kernel);

        let (_, ret) = conv_2d_fft_inner(data_ext, kernel_ext);

        let out_size = [
            (data_size[0] - kernel_size[0] + padding_size.pad[0].iter().sum::<usize>()) + 1,
            (data_size[1] - kernel_size[1] + padding_size.pad[1].iter().sum::<usize>()) + 1,
        ];

        ret.slice(s!(
           kernel_size[0] - 1..kernel_size[0] - 1 + out_size[0];padding_size.stride[0],
           kernel_size[1] - 1..kernel_size[1] - 1 + out_size[1];padding_size.stride[1],
        ))
        .to_owned()
        .into()
    }
}

fn get_good_fft_size(size: &[usize; N]) -> [usize; N] {
    [good_size_cc(size[0]), good_size_rr(size[1])]
}

pub(crate) fn pad_fft<S, T>(
    data: &ArrayBase<S, Ix2>,
    padding_size: &ExplicitPadding<N>,
    padding_mode: &PaddingMode<N, T>,
) -> Option<Array2<T>>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let data_size = std::array::from_fn(|i| data.shape()[i]);
    let good_size = get_good_fft_size(&data_size);
    let padded_size = [
        data_size[0] + padding_size.pad[0].iter().sum::<usize>(),
        data_size[1] + padding_size.pad[1].iter().sum::<usize>(),
    ];

    let fft_size = if good_size[0] > padded_size[0] || good_size[1] > padded_size[1] {
        good_size
    } else {
        get_good_fft_size(&padded_size)
    };

    match *padding_mode {
        PaddingMode::Zeros => pad_const_fft(data, fft_size, padded_size, padding_size, T::zero()),
        PaddingMode::Const(value) => {
            pad_const_fft(data, fft_size, padded_size, padding_size, value)
        }
        _ => {
            let mut buf = Array2::from_elem(fft_size, T::zero());
            let buf_slice = buf.slice_mut(s!(..padded_size[0], ..padded_size[1]));
            let buf_slice = padding::pad_const_inner(data, buf_slice, padding_size);
            padding::pad_inner(data, buf_slice, padding_size, padding_mode)?;
            buf
        }
    }
    .into()
}

fn pad_const_fft<S, T>(
    data: &ArrayBase<S, Ix2>,
    fft_size: [usize; N],
    padded_size: [usize; N],
    padding_size: &ExplicitPadding<N>,
    value: T,
) -> Array2<T>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let mut buf = Array2::from_elem(fft_size, value);
    let buf_slice = buf.slice_mut(s!(..padded_size[0], ..padded_size[1]));
    let _ = padding::pad_const_inner(data, buf_slice, padding_size);

    buf
}

fn conv_2d_fft_inner<S, T>(
    mut data_ext: ArrayBase<S, Ix2>,
    mut kernel_ext: ArrayBase<S, Ix2>,
) -> ((usize, usize), Array2<T>)
where
    S: DataMut<Elem = T>,
    T: rustfft::FftNum + Float,
{
    // let output_shape = (
    //     data.shape()[0] + kernel.shape()[0] - 1,
    //     data.shape()[1] + kernel.shape()[1] - 1,
    // );

    let output_shape = (data_ext.shape()[0], data_ext.shape()[1]);

    let fft_shape = output_shape;

    // let fft_shape = (good_size_cc(output_shape.0), good_size_rr(output_shape.1));

    // let fft_shape = (good_size_c(output_shape.0), good_size_r(output_shape.1));
    // let fft_shape_23 = (good_size_cc(output_shape.0), good_size_rr(output_shape.1));

    // let fft_shape = (
    //     if fft_shape_23.0 - fft_shape.0 < fft_shape.0 / 10 {
    //         fft_shape_23.0
    //     } else {
    //         fft_shape.0
    //     },
    //     if fft_shape_23.1 - fft_shape.1 < fft_shape.1 / 10 {
    //         fft_shape_23.1
    //     } else {
    //         fft_shape.1
    //     },
    // );

    // let mut data_ext = Array2::zeros(fft_shape);
    // data_ext
    //     .slice_mut(s!(..data.shape()[0], ..data.shape()[1]))
    //     .assign(data);

    let mut rp = realfft::RealFftPlanner::new();
    let mut cp = rustfft::FftPlanner::new();

    let data_spec = fft_2d::forward(&mut data_ext, &mut rp, &mut cp);

    // let kernel = Array2::from_shape_vec(
    //     kernel.dim(),
    //     kernel.as_slice().unwrap().iter().rev().copied().collect(),
    // )
    // .unwrap();
    // let mut kernel_ext = Array2::zeros(fft_shape);
    // kernel_ext
    //     .slice_mut(s![..kernel.shape()[0], ..kernel.shape()[1]])
    //     .assign(&kernel);
    let kernel_spec = fft_2d::forward(&mut kernel_ext, &mut rp, &mut cp);

    let mut mul_spec = data_spec * kernel_spec;

    (
        output_shape,
        fft_2d::inverse(&mut mul_spec, fft_shape.1, &mut rp, &mut cp),
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
    loop {
        let new_fac = best_fac / 6 * 5;
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
    let res = n % 2;
    let n = n / 2;

    (good_size_cc(n)) * 2 + res
}

pub fn good_size_c(n: usize) -> usize {
    if n <= 12 {
        return n;
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
        let kernel = array![[1, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0],];

        dbg!(input_pixels
            .mapv(|x| x as f64)
            .conv_2d_fft(
                &kernel.mapv(|x| x as f64),
                PaddingSize::Same,
                PaddingMode::Zeros
            )
            .unwrap()
            .mapv(|x| x.round() as i32));
        // dbg!(&conv_2d_c2c::<
        //     f64,
        //     ndarray::OwnedRepr<f64>,
        //     ndarray::OwnedRepr<f64>,
        // >(&input_pixels.mapv(|x| x as f64), &kernel.mapv(|x| x as f64)));
    }

    #[test]
    fn test_same_conv() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        let output_pixels = array![
            [2, 2, 3, 1, 1],
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [0, 2, 2, 1, 1],
        ];

        // let conv_type = PaddingSize::Same.unfold(&[kernel.shape()[0], kernel.shape()[1]]);

        // let (pad_input_size, out_size) = crate::conv_2d::padding::get_size(
        //     &[input_pixels.shape()[0], input_pixels.shape()[1]],
        //     &[kernel.shape()[0], kernel.shape()[1]],
        //     &conv_type,
        // );

        let ret = input_pixels
            .mapv(f64::from)
            .conv_2d_fft(
                &kernel.mapv(f64::from),
                PaddingSize::Valid,
                PaddingMode::Custom([BorderType::Reflect, BorderType::Circular]),
            )
            .unwrap();

        // let input_pixels = pad_fft(&input_pixels, &conv_type, &PaddingMode::Replicate).unwrap();
        // let input_pixels =
        //     crate::conv_2d::padding::pad(&input_pixels, &conv_type, &PaddingMode::Zeros).unwrap();

        dbg!(&input_pixels);

        // let (_, ret) =
        //     conv_2d_fft_inner(&dbg!(input_pixels.mapv(f64::from)), &kernel.mapv(f64::from));
        // dbg!(&ret);
        dbg!(&ret.mapv(|x| x.round() as i32));
    }

    #[test]
    fn torch_same_output_pixels() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1], [2, -1, 3]];

        let torch_same_output_pixels = array![
            [4, 4, 5, 2, 2],
            [2, 5, 5, 5, 2],
            [1, 7, 5, 5, 3],
            [4, 4, 5, 5, 3],
            [1, 2, 3, 4, 1]
        ];
        assert_eq!(
            input_pixels
                .mapv(f64::from)
                .conv_2d_fft(
                    &kernel.mapv(f64::from),
                    PaddingSize::Same,
                    PaddingMode::Zeros
                )
                .unwrap()
                .mapv(|x| x.round() as i32),
            torch_same_output_pixels
        );
    }

    #[test]
    fn custom_conv_with_pad_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1], [2, -1, 3]];

        let padding = [2, 0];
        let stride = [1, 1];
        let custom_conv_with_pad_output_pixels = array![
            [4, 5, 2],
            [5, 5, 5],
            [7, 5, 5],
            [4, 5, 5],
            [2, 3, 4],
            [2, 2, 1]
        ];

        assert_eq!(
            input_pixels
                .mapv(f64::from)
                .conv_2d_fft(
                    &kernel.mapv(f64::from),
                    PaddingSize::Custom(padding, stride),
                    PaddingMode::Zeros
                )
                .unwrap()
                .mapv(|x| x.round() as i32),
            custom_conv_with_pad_output_pixels
        );
    }

    #[test]
    fn test_best_fft_len() {
        // let mut n = 93059;
        let mut n = 5000;
        dbg!(good_size_c(n));
        dbg!(good_size_cc(n));
        dbg!(good_size_r(n));
        dbg!(good_size_rr(n));
    }
}
