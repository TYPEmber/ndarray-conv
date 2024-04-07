use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, Data, Dim, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo, SliceInfoElem,
};
use num::{traits::NumAssign, Float};
use rustfft::FftNum;

use crate::{dilation::IntoKernelWithDilation, ConvMode, PaddingMode};

mod fft;
mod good_size;
mod padding;

pub trait ConvFFTExt<'a, T: FftNum + NumAssign, S: ndarray::RawData, const N: usize> {
    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, S, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>>;
}

impl<'a, T: NumAssign + Copy, S: ndarray::RawData, const N: usize> ConvFFTExt<'a, T, S, N>
    for ArrayBase<S, Dim<[Ix; N]>>
where
    T: NumAssign + Debug + FftNum,
    S: Data<Elem = T> + 'a,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, S, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>> {
        let kwd = kernel.into_kernel_with_dilation();

        let cm = conv_mode.unfold(&kwd);

        let data_raw_dim = self.raw_dim();
        let kernel_raw_dim = kwd.kernel.raw_dim();
        let kernel_raw_dim_with_dilation: [usize; N] =
            std::array::from_fn(|i| kernel_raw_dim[i] * kwd.dilation[i] - kwd.dilation[i] + 1);
        let fft_size = good_size::compute::<N>(&std::array::from_fn(|i| {
            (data_raw_dim[i] + cm.padding[i][0] + cm.padding[i][1])
                .max(kernel_raw_dim_with_dilation[i])
        }));

        let mut data_pd = padding::data(self, padding_mode, cm.padding, fft_size);
        let mut kernel_pd = padding::kernel(kwd, fft_size);

        let mut fft = fft::Processor::default();

        let data_pd_fft = fft.forward(&mut data_pd);
        let kernel_pd_fft = fft.forward(&mut kernel_pd);

        let mul_spec = data_pd_fft.mul(kernel_pd_fft);

        use std::ops::Mul;

        let output = fft.backward(mul_spec);

        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{dilation::WithDilation, ConvExt};

    use super::*;

    #[test]
    fn conv_fft() {
        let arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        let kernel = array![
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ];

        let res_normal = arr
            .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();

        dbg!(res_normal);

        let res_fft = arr
            .map(|&x| x as f32)
            .conv_fft(
                &kernel.map(|&x| x as f32),
                ConvMode::Full,
                PaddingMode::Zeros,
            )
            .unwrap()
            .map(|x| x.round() as i32);
        dbg!(res_fft);

        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1], [1, 1]];

        let res_normal = arr
            .conv(kernel.with_dilation(2), ConvMode::Full, PaddingMode::Zeros)
            .unwrap();
        dbg!(res_normal);

        let res_fft = arr
            .map(|&x| x as f32)
            .conv_fft(
                kernel.map(|&x| x as f32).with_dilation(2),
                ConvMode::Full,
                PaddingMode::Zeros,
            )
            .unwrap()
            .map(|x| x.round() as i32);
        dbg!(res_fft);

        // assert_eq!(res_normal, res_fft);

        let arr = array![1, 2, 3, 4, 5, 6];
        let kernel = array![1, 1, 1];

        let res_normal = arr
            .conv(
                kernel.with_dilation(2),
                ConvMode::Custom {
                    padding: [4],
                    strides: [1],
                },
                PaddingMode::Zeros,
            )
            .unwrap();
        dbg!(&res_normal);

        let res_fft = arr
            .map(|&x| x as f32)
            .conv_fft(
                kernel.map(|&x| x as f32).with_dilation(2),
                ConvMode::Custom {
                    padding: [7],
                    strides: [1],
                },
                PaddingMode::Zeros,
            )
            .unwrap()
            .map(|x| x.round() as i32);
        dbg!(res_fft);
    }
}
