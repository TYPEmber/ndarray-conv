use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, Data, Dim, IntoDimension, Ix, RawData, RemoveAxis, SliceArg,
    SliceInfo, SliceInfoElem,
};
use num::traits::NumAssign;
use rustfft::FftNum;

use crate::{dilation::IntoKernelWithDilation, ConvMode, PaddingMode};

mod fft;
mod good_size;
mod padding;

pub trait ConvFFTExt<'a, T, S, SK, const N: usize>
where
    T: FftNum + NumAssign,
    S: RawData,
    SK: RawData,
{
    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Result<Array<T, Dim<[Ix; N]>>, crate::Error<N>>;
}

impl<'a, T, S, SK, const N: usize> ConvFFTExt<'a, T, S, SK, N> for ArrayBase<S, Dim<[Ix; N]>>
where
    T: NumAssign + Debug + FftNum,
    S: Data<Elem = T> + 'a,
    SK: Data<Elem = T> + 'a,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Result<Array<T, Dim<[Ix; N]>>, crate::Error<N>> {
        let kwd = kernel.into_kernel_with_dilation();

        let data_raw_dim = self.raw_dim();
        if self.shape().iter().product::<usize>() == 0 {
            return Err(crate::Error::DataShape(data_raw_dim));
        }

        let kernel_raw_dim = kwd.kernel.raw_dim();
        if kwd.kernel.shape().iter().product::<usize>() == 0 {
            return Err(crate::Error::DataShape(kernel_raw_dim));
        }

        let kernel_raw_dim_with_dilation: [usize; N] =
            std::array::from_fn(|i| kernel_raw_dim[i] * kwd.dilation[i] - kwd.dilation[i] + 1);

        let cm = conv_mode.unfold(&kwd);

        let pds_raw_dim: [usize; N] =
            std::array::from_fn(|i| (data_raw_dim[i] + cm.padding[i][0] + cm.padding[i][1]));
        if !(0..N).all(|i| kernel_raw_dim_with_dilation[i] <= pds_raw_dim[i]) {
            return Err(crate::Error::MismatchShape(
                conv_mode,
                kernel_raw_dim_with_dilation,
            ));
        }

        let fft_size = good_size::compute::<N>(&std::array::from_fn(|i| {
            pds_raw_dim[i].max(kernel_raw_dim_with_dilation[i])
        }));

        let mut data_pd = padding::data(self, padding_mode, cm.padding, fft_size);
        let mut kernel_pd = padding::kernel(kwd, fft_size);

        let mut fft = fft::Processor::default();

        let mut data_pd_fft = fft.forward(&mut data_pd);
        let kernel_pd_fft = fft.forward(&mut kernel_pd);

        data_pd_fft.zip_mut_with(&kernel_pd_fft, |d, k| *d *= *k);
        // let mul_spec = data_pd_fft * kernel_pd_fft;

        let output = fft.backward(data_pd_fft);

        let output = output.slice_move(unsafe {
            SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
                start: kernel_raw_dim_with_dilation[i] as isize - 1,
                end: Some((pds_raw_dim[i]) as isize),
                step: cm.strides[i] as isize,
            }))
            .unwrap()
        });

        Ok(output)
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
        // dbg!(res_normal);

        let res_fft = arr
            .map(|&x| x as f32)
            .conv_fft(
                &kernel.map(|&x| x as f32),
                ConvMode::Same,
                PaddingMode::Zeros,
            )
            .unwrap()
            .map(|x| x.round() as i32);
        // dbg!(res_fft);

        assert_eq!(res_normal, res_fft);

        //

        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 0], [3, 1]];

        let res_normal = arr
            .conv(
                kernel.with_dilation(2),
                ConvMode::Custom {
                    padding: [3, 3],
                    strides: [2, 2],
                },
                PaddingMode::Replicate,
            )
            .unwrap();
        // dbg!(res_normal);

        let res_fft = arr
            .map(|&x| x as f64)
            .conv_fft(
                kernel.map(|&x| x as f64).with_dilation(2),
                ConvMode::Custom {
                    padding: [3, 3],
                    strides: [2, 2],
                },
                PaddingMode::Replicate,
            )
            .unwrap()
            .map(|x| x.round() as i32);
        // dbg!(res_fft);

        assert_eq!(res_normal, res_fft);

        //

        let arr = array![1, 2, 3, 4, 5, 6];
        let kernel = array![1, 1, 1, 1];

        let res_normal = arr
            .conv(kernel.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        // dbg!(&res_normal);

        let res_fft = arr
            .map(|&x| x as f32)
            .conv_fft(
                kernel.map(|&x| x as f32).with_dilation(2),
                ConvMode::Same,
                PaddingMode::Zeros,
            )
            .unwrap()
            .map(|x| x.round() as i32);
        // dbg!(res_fft);

        assert_eq!(res_normal, res_fft);
    }
}
