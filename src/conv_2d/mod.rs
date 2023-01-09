use std::ops::Mul;

use ndarray::{prelude::*, OwnedRepr};
use num::traits::{AsPrimitive, FromPrimitive, NumAssign};

pub mod fft;
mod fft_2d;

pub mod padding;

use crate::{BorderType, PaddingMode};

use super::ConvType;

pub trait Conv2DExt<T: NumAssign + Copy, S: ndarray::Data> {
    fn conv_2d(
        &self,
        kernel: &ArrayBase<S, Ix2>,
        conv_type: ConvType<2>,
        padding_mode: PaddingMode<2, T>,
    ) -> Option<Array2<T>>;
}

impl<T, S> Conv2DExt<T, S> for ArrayBase<S, Ix2>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    fn conv_2d(
        &self,
        kernel: &ArrayBase<S, Ix2>,
        conv_type: ConvType<2>,
        padding_mode: PaddingMode<2, T>,
    ) -> Option<Array2<T>> {
        // let (h, w) = (self.shape()[0], self.shape()[1]);
        let input_size = [self.shape()[0], self.shape()[1]];
        let kernel_size = [kernel.shape()[0], kernel.shape()[1]];
        let (pad_input_size, padding, stride) =
            padding::get_size(&input_size, &kernel_size, conv_type);
        // dbg!(&pad_input_size, &padding, &stride, &input_size);
        conv_2d_inner(
            self,
            kernel,
            &input_size,
            &kernel_size,
            &padding,
            &stride,
            &pad_input_size,
            padding_mode,
        )
    }
}

fn conv_2d_inner<S, T>(
    data: &ArrayBase<S, Ix2>,
    kernel: &ArrayBase<S, Ix2>,
    input_size: &[usize; 2],
    kernel_size: &[usize; 2],
    padding: &[[usize; 2]; 2],
    stride: &[usize; 2],
    pad_input_size: &[usize; 2],
    padding_mode: PaddingMode<2, T>,
) -> Option<Array2<T>>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let (input_h, input_w) = (input_size[0], input_size[1]);
    let (kernel_h, kernel_w) = (kernel_size[0], kernel_size[1]);
    let [padding_h, padding_w] = *padding;
    let [stride_h, stride_w] = *stride;
    let (pad_input_h, pad_input_w) = (pad_input_size[0], pad_input_size[1]);

    let (out_h, out_w) = (
        (input_h - kernel_h + padding_h.iter().sum::<usize>()) / stride_h + 1,
        (input_w - kernel_w + padding_w.iter().sum::<usize>()) / stride_w + 1,
    );

    // padding
    let mut pad_input = padding::pad(data, padding, padding, pad_input_size, padding_mode);
    // let mut pad_input = Array2::zeros((pad_input_h, pad_input_w));
    // let mut pad_input = pad(data, &padding, &[pad_input_h, pad_input_w], pad_input, padding_mode);

    dbg!(&pad_input);
    // let mut pad_input = Array2::zeros((pad_input_h, pad_input_w));
    // let mut sub_pad_input = pad_input.slice_mut(s!(
    //     padding_h[0]..input_h + padding_h[0],
    //     padding_w[0]..input_w + padding_w[0]
    // ));
    // sub_pad_input.assign(data);

    // let mut ret = Array::<T, Ix2>::zeros((out_h, out_w));

    // let mut offset = vec![];
    // for r in -(kernel.shape()[0] as isize / 2)
    //     ..=kernel_h as isize / 2 - if kernel_h % 2 == 0 { 1 } else { 0 }
    // {
    //     for c in
    //         -(kernel_w as isize / 2)..=kernel_w as isize / 2 - if kernel_w % 2 == 0 { 1 } else { 0 }
    //     {
    //         let k = kernel[[
    //             (r + kernel_h as isize / 2) as usize,
    //             (c + kernel_w as isize / 2) as usize,
    //         ]];
    //         if k != T::zero() {
    //             offset.push((r * pad_input_w as isize + c, k));
    //         }
    //     }
    // }

    // // let (tmp_h, tmp_w) = ((input_h - out_h) / 2, (input_w - out_w) / 2);
    // let valid_input = pad_input.slice(s!(kernel_h / 2..pad_input_h - kernel_h / 2;stride_h, kernel_w / 2..pad_input_w - kernel_w / 2;stride_w));
    // unsafe {
    //     ndarray::Zip::from(&mut ret)
    //         .and(&valid_input)
    //         .for_each(|r, s| {
    //             let mut temp = T::zero();
    //             for (o, k) in offset.iter() {
    //                 temp += (*(s as *const T).offset(*o)) * *k
    //             }
    //             *r = temp
    //         });
    // }
    // Some(ret)

    let mut ret1 = Array2::zeros((1, out_h * out_w));

    let mut offsets = vec![];
    for i in 0..kernel_h {
        for j in 0..kernel_w {
            if kernel[(i, j)] == T::zero() {
                continue;
            }
            offsets.push((i * pad_input_w + j, kernel[(i, j)]));
        }
    }
    // dbg!(input_offset);

    let mut ret_idx = 0;
    unsafe {
        for i in (0..=pad_input_h - kernel_h).step_by(stride_h) {
            for j in (0..=pad_input_w - kernel_w).step_by(stride_w) {
                let mut tmp_res = T::zero();
                let cur = pad_input.uget((i, j));

                offsets.iter().for_each(|(tmp_offset, tmp_kernel)| {
                    tmp_res += *(cur as *const T).add(*tmp_offset) * *tmp_kernel
                });

                *ret1.uget_mut((0, ret_idx)) = tmp_res;
                ret_idx += 1;
            }
        }
    }
    Some(ret1.into_shape((out_h, out_w)).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_conv_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        // FULL OUTPUT
        let full_output_pixels = array![
            [1, 1, 2, 1, 1, 0, 0],
            [0, 2, 2, 3, 1, 1, 0],
            [1, 1, 4, 3, 4, 1, 1],
            [0, 1, 2, 4, 3, 3, 0],
            [0, 1, 2, 3, 4, 1, 1],
            [0, 0, 2, 2, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0]
        ];
        assert_eq!(
            input_pixels
                .conv_2d(&kernel, ConvType::Full, PaddingMode::Zeros)
                .unwrap(),
            full_output_pixels
        );
    }

    #[test]
    fn same_conv_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        // SAME OUTPUT
        let same_output_pixels = array![
            [2, 2, 3, 1, 1],
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [0, 2, 2, 1, 1],
        ];
        assert_eq!(
            input_pixels
                .conv_2d(&kernel, ConvType::Same, PaddingMode::Zeros)
                .unwrap(),
            same_output_pixels
        );
    }

    #[test]
    fn valid_conv_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        // VALID OUTPUT
        let valid_output_pixels = array![[4, 3, 4], [2, 4, 3], [2, 3, 4]];
        assert_eq!(
            input_pixels
                .conv_2d(&kernel, ConvType::Valid, PaddingMode::Zeros)
                .unwrap(),
            valid_output_pixels
        );
    }

    #[test]
    fn custom_conv_with_pad_test() {
        // let input_pixels = array![
        //     [1, 1, 1, 0, 0],
        //     [0, 1, 1, 1, 0],
        //     [0, 0, 1, 1, 1],
        //     [0, 0, 1, 1, 0],
        //     [0, 1, 1, 0, 0],
        // ];
        let input_pixels = array![
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1], [2, -1, 3]];

        let padding = [2, 1];
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
                .conv_2d(
                    &kernel,
                    ConvType::Custom(padding, stride),
                    PaddingMode::Warp
                )
                .unwrap(),
            custom_conv_with_pad_output_pixels
        );
    }

    #[test]
    fn custom_conv_with_pad_stride_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        let padding = [3, 2];
        let stride = [3, 3];
        let custom_conv_with_pad_stride_output_pixels = array![[0, 0, 0], [1, 3, 1], [0, 2, 0]];
        assert_eq!(
            input_pixels
                .conv_2d(
                    &kernel,
                    ConvType::Custom(padding, stride),
                    PaddingMode::Zeros
                )
                .unwrap(),
            custom_conv_with_pad_stride_output_pixels
        );
    }

    #[test]
    fn opencv_same_conv_test() {
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
                .conv_2d(&kernel, ConvType::Same, PaddingMode::Zeros)
                .unwrap(),
            torch_same_output_pixels
        );
    }

    #[test]
    fn torch_valid_conv_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1], [2, -1, 3]];

        let torch_valid_output_pixels = array![[7, 5, 5], [4, 5, 5]];
        assert_eq!(
            input_pixels
                .conv_2d(&kernel, ConvType::Valid, PaddingMode::Zeros)
                .unwrap(),
            torch_valid_output_pixels
        );
    }

    #[test]
    fn test_stride() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let a = input_pixels.slice(s!(..,..; 2));

        dbg!(a);
    }
}
