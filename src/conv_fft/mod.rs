use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, ArrayView, Data, Dim, Dimension, IntoDimension, Ix, RemoveAxis, SliceArg,
    SliceInfo, SliceInfoElem,
};
use num::{traits::NumAssign, Float};

use crate::{dilation::IntoKernelWithDilation, ConvMode, PaddingMode};

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
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, S, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>> {
        todo!()
    }
}
