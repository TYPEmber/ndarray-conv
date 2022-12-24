use ndarray::prelude::*;
use num::traits::{AsPrimitive, FromPrimitive, NumAssign};

pub mod c2cfft;
pub mod fft;
mod fft_2d;
pub mod naive_conv;
pub mod ndrustfft;

use super::ConvType;

pub trait Conv2DExt<T: NumAssign + Copy, S: ndarray::Data> {
    fn conv_2d(&self, kernel: &ArrayBase<S, Ix2>, conv_type: ConvType<2>) -> Option<Array2<T>>;
}

impl<T, S> Conv2DExt<T, S> for ArrayBase<S, Ix2>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign,
{
    fn conv_2d(&self, kernel: &ArrayBase<S, Ix2>, conv_type: ConvType<2>) -> Option<Array2<T>> {
        let arr = [0; 5];

        // conv with same size output
        let (h, w) = (self.shape()[0], self.shape()[1]);
        let (kernel_h, kernel_w) = (kernel.shape()[0], kernel.shape()[1]);
        let (stride_h, stride_w) = (1, 1);
        let (pad_h, pad_w) = (
            (h * stride_h - h + kernel_h - 1) / 2,
            (w * stride_w - w + kernel_w - 1) / 2,
        );
        let (new_h, new_w) = (h + 2 * pad_h, w + 2 * pad_w);

        let mut pad_input = Array2::zeros((new_h, new_w));
        let mut sub_pad_input = pad_input.slice_mut(s!(pad_h..h + pad_h, pad_w..w + pad_w));
        sub_pad_input.assign(self);

        let mut ret = Array::<T, Ix2>::zeros(self.dim());

        let mut offset = vec![];
        for r in -(kernel.shape()[0] as isize / 2)
            ..=kernel_h as isize / 2 - if kernel_h % 2 == 0 { 1 } else { 0 }
        {
            for c in -(kernel_w as isize / 2)
                ..=kernel_w as isize / 2 - if kernel_w % 2 == 0 { 1 } else { 0 }
            {
                let k = kernel[[
                    (r + kernel_h as isize / 2) as usize,
                    (c + kernel_w as isize / 2) as usize,
                ]];
                if k != T::zero() {
                    offset.push((r * new_w as isize + c, k));
                }
            }
        }

        unsafe {
            ndarray::Zip::from(&mut ret)
                .and(&sub_pad_input)
                .for_each(|r, s| {
                    let mut temp = T::zero();
                    for (o, k) in offset.iter() {
                        temp += (*(s as *const T).offset(*o)) * *k
                    }
                    *r = temp
                });
        }
        Some(ret)
    }
}

#[cfg(test)]
mod tests {
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


        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];
        // let kernel = array![[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0]];
        dbg!(&kernel, &kernel.shape(), &kernel.dim());

        assert_ne!(dbg!(input_pixels.conv_2d(&kernel, ConvType::Same)), None);
        // assert_eq!(dbg!(input_pixels.conv_2d(&kernel)).unwrap(), output_pixels);
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
