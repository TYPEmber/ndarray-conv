use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, Data, Dim, Dimension, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo, SliceInfoElem, StrideShape
};
use num::{traits::NumAssign, Float};

use crate::{dilation::IntoKernelWithDilation, ConvMode, PaddingMode};

mod good_size;
mod padding;

pub trait ConvFFTExt<'a, T: Float + NumAssign, S: ndarray::RawData, const N: usize> {
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
    T: Float + NumAssign + Debug,
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

        let mut cm = conv_mode.unfold(&kwd);

        let data_raw_dim = self.raw_dim();
        let kernel_raw_dim = kwd.kernel.raw_dim();
        let kernel_raw_dim_with_dilation: [usize; N] =
            std::array::from_fn(|i| kernel_raw_dim[i] * kwd.dilation[i] - kwd.dilation[i] + 1);
        let fft_size = good_size::compute::<N>(&std::array::from_fn(|i| {
            (data_raw_dim[i] + cm.padding[i][0] + cm.padding[i][1])
                .max(kernel_raw_dim_with_dilation[i])
        }));

        cm.padding
            .iter_mut()
            .zip(fft_size)
            .zip(self.shape())
            .for_each(|((mut pd, fft_size), &raw_dim)| pd[1] += fft_size - raw_dim);

        let data_pd = padding::data(self, padding_mode, cm.padding, fft_size);
        let kernel_pd = padding::kernel(kwd, fft_size);

        todo!()
    }
}
