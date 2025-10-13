//! Provides FFT-accelerated convolution operations.
//!
//! This module offers the `ConvFFTExt` trait, which extends `ndarray`
//! with FFT-based convolution methods.

use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, Data, Dim, IntoDimension, Ix, RawData, RemoveAxis, SliceArg, SliceInfo,
    SliceInfoElem,
};
use num::traits::NumAssign;
use rustfft::FftNum;

use crate::{dilation::IntoKernelWithDilation, ConvMode, PaddingMode};

mod good_size;
mod padding;
mod processor;

// pub use fft::Processor;
pub use processor::{get as get_processor, GetProcessor, Processor};

// /// Represents a "baked" convolution operation.
// ///
// /// This struct holds pre-computed data for performing FFT-accelerated
// /// convolutions, including the FFT size, FFT processor, scratch space,
// /// and padding information. It's designed to optimize repeated
// /// convolutions with the same kernel and settings.
// pub struct Baked<T, SK, const N: usize>
// where
//     T: NumAssign + Debug + Copy,
//     SK: RawData,
// {
//     fft_size: [usize; N],
//     fft_processor: impl Processor<T>,
//     scratch: Vec<Complex<T>>,
//     cm: ExplicitConv<N>,
//     padding_mode: PaddingMode<N, T>,
//     kernel_raw_dim_with_dilation: [usize; N],
//     pds_raw_dim: [usize; N],
//     kernel_pd: Array<T, Dim<[Ix; N]>>,
//     _sk_hint: PhantomData<SK>,
// }

/// Extends `ndarray`'s `ArrayBase` with FFT-accelerated convolution operations.
///
/// This trait adds the `conv_fft` and `conv_fft_with_processor` methods to `ArrayBase`,
/// enabling efficient FFT-based convolutions on N-dimensional arrays.
///
/// # Type Parameters
///
/// *   `T`: The numeric type used internally for FFT operations. Must be a floating-point type that implements `FftNum`.
/// *   `InElem`: The element type of the input arrays. Can be real (`T`) or complex (`Complex<T>`).
/// *   `S`: The data storage type of the input array.
/// *   `SK`: The data storage type of the kernel array.
///
/// # Methods
///
/// *   `conv_fft`: Performs an FFT-accelerated convolution with default settings.
/// *   `conv_fft_with_processor`: Performs an FFT-accelerated convolution using a provided `Processor` instance, allowing for reuse of FFT plans across multiple convolutions for better performance.
///
/// # Example
///
/// ```rust
/// use ndarray::prelude::*;
/// use ndarray_conv::{ConvFFTExt, ConvMode, PaddingMode};
///
/// let arr = array![[1., 2.], [3., 4.]];
/// let kernel = array![[1., 0.], [0., 1.]];
/// let result = arr.conv_fft(&kernel, ConvMode::Same, PaddingMode::Zeros).unwrap();
/// ```
///
/// # Notes
///
/// FFT-based convolutions are generally faster for larger kernels but may have higher overhead for smaller kernels.
/// Use standard convolution (`ConvExt::conv`) for small kernels or when working with integer types.
///
/// # Performance Tips
///
/// For repeated convolutions with different data but the same kernel and settings, consider using
/// `conv_fft_with_processor` to reuse the FFT planner and avoid redundant setup overhead.
pub trait ConvFFTExt<'a, T, InElem, S, SK, const N: usize>
where
    T: NumAssign + Copy + FftNum,
    InElem: processor::GetProcessor<T, InElem> + Copy + NumAssign,
    S: RawData,
    SK: RawData,
{
    /// Performs an FFT-accelerated convolution operation.
    ///
    /// This method convolves the input array with a given kernel using FFT,
    /// which is typically faster for larger kernels.
    ///
    /// # Arguments
    ///
    /// * `kernel`: The convolution kernel. Can be a reference to an array, or an array with dilation settings.
    /// * `conv_mode`: The convolution mode (`Full`, `Same`, `Valid`, `Custom`, `Explicit`).
    /// * `padding_mode`: The padding mode (`Zeros`, `Const`, `Reflect`, `Replicate`, `Circular`, `Custom`, `Explicit`).
    ///
    /// # Returns
    ///
    /// Returns `Ok(Array<InElem, Dim<[Ix; N]>>)` containing the convolution result, or an `Err(Error<N>)` if the operation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use ndarray_conv::{ConvFFTExt, ConvMode, PaddingMode};
    ///
    /// let input = array![[1.0, 2.0], [3.0, 4.0]];
    /// let kernel = array![[1.0, 0.0], [0.0, 1.0]];
    /// let result = input.conv_fft(&kernel, ConvMode::Same, PaddingMode::Zeros).unwrap();
    /// ```
    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, InElem>,
    ) -> Result<Array<InElem, Dim<[Ix; N]>>, crate::Error<N>>;

    /// Performs an FFT-accelerated convolution using a provided processor.
    ///
    /// This method is useful when performing multiple convolutions, as it allows
    /// reusing the FFT planner and avoiding redundant initialization overhead.
    ///
    /// # Arguments
    ///
    /// * `kernel`: The convolution kernel.
    /// * `conv_mode`: The convolution mode.
    /// * `padding_mode`: The padding mode.
    /// * `fft_processor`: A mutable reference to an FFT processor instance.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Array<InElem, Dim<[Ix; N]>>)` containing the convolution result, or an `Err(Error<N>)` if the operation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::array;
    /// use ndarray_conv::{ConvFFTExt, ConvMode, PaddingMode, get_fft_processor};
    ///
    /// let input1 = array![[1.0, 2.0], [3.0, 4.0]];
    /// let input2 = array![[5.0, 6.0], [7.0, 8.0]];
    /// let kernel = array![[1.0, 0.0], [0.0, 1.0]];
    ///
    /// // Reuse the same processor for multiple convolutions
    /// let mut proc = get_fft_processor::<f32, f32>();
    /// let result1 = input1.conv_fft_with_processor(&kernel, ConvMode::Same, PaddingMode::Zeros, &mut proc).unwrap();
    /// let result2 = input2.conv_fft_with_processor(&kernel, ConvMode::Same, PaddingMode::Zeros, &mut proc).unwrap();
    /// ```
    fn conv_fft_with_processor(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, InElem>,
        fft_processor: &mut impl Processor<T, InElem>,
    ) -> Result<Array<InElem, Dim<[Ix; N]>>, crate::Error<N>>;

    // fn conv_fft_bake(
    //     &self,
    //     kernel: impl IntoKernelWithDilation<'a, SK, N>,
    //     conv_mode: ConvMode<N>,
    //     padding_mode: PaddingMode<N, T>,
    // ) -> Result<Baked<T, SK, N>, crate::Error<N>>;

    // fn conv_fft_with_baked(&self, baked: &mut Baked<T, SK, N>) -> Array<T, Dim<[Ix; N]>>;
}

impl<'a, T, InElem, S, SK, const N: usize> ConvFFTExt<'a, T, InElem, S, SK, N>
    for ArrayBase<S, Dim<[Ix; N]>>
where
    T: NumAssign + FftNum,
    InElem: processor::GetProcessor<T, InElem> + NumAssign + Copy + Debug,
    S: Data<Elem = InElem> + 'a,
    SK: Data<Elem = InElem> + 'a,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    // fn conv_fft_bake(
    //     &self,
    //     kernel: impl IntoKernelWithDilation<'a, SK, N>,
    //     conv_mode: ConvMode<N>,
    //     padding_mode: PaddingMode<N, T>,
    // ) -> Result<Baked<T, SK, N>, crate::Error<N>> {
    //     let mut fft_processor = Processor::default();

    //     let kwd = kernel.into_kernel_with_dilation();

    //     let data_raw_dim = self.raw_dim();
    //     if self.shape().iter().product::<usize>() == 0 {
    //         return Err(crate::Error::DataShape(data_raw_dim));
    //     }

    //     let kernel_raw_dim = kwd.kernel.raw_dim();
    //     if kwd.kernel.shape().iter().product::<usize>() == 0 {
    //         return Err(crate::Error::DataShape(kernel_raw_dim));
    //     }

    //     let kernel_raw_dim_with_dilation: [usize; N] =
    //         std::array::from_fn(|i| kernel_raw_dim[i] * kwd.dilation[i] - kwd.dilation[i] + 1);

    //     let cm = conv_mode.unfold(&kwd);

    //     let pds_raw_dim: [usize; N] =
    //         std::array::from_fn(|i| (data_raw_dim[i] + cm.padding[i][0] + cm.padding[i][1]));
    //     if !(0..N).all(|i| kernel_raw_dim_with_dilation[i] <= pds_raw_dim[i]) {
    //         return Err(crate::Error::MismatchShape(
    //             conv_mode,
    //             kernel_raw_dim_with_dilation,
    //         ));
    //     }

    //     let fft_size = good_size::compute::<N>(&std::array::from_fn(|i| {
    //         pds_raw_dim[i].max(kernel_raw_dim_with_dilation[i])
    //     }));

    //     let scratch = fft_processor.get_scratch(fft_size);

    //     let kernel_pd = padding::kernel(kwd, fft_size);

    //     Ok(Baked {
    //         fft_size,
    //         fft_processor,
    //         scratch,
    //         cm,
    //         padding_mode,
    //         kernel_raw_dim_with_dilation,
    //         pds_raw_dim,
    //         kernel_pd,
    //         _sk_hint: PhantomData,
    //     })
    // }

    // fn conv_fft_with_baked(&self, baked: &mut Baked<T, SK, N>) -> Array<T, Dim<[Ix; N]>> {
    //     let Baked {
    //         scratch,
    //         fft_processor,
    //         fft_size,
    //         cm,
    //         padding_mode,
    //         kernel_pd,
    //         kernel_raw_dim_with_dilation,
    //         pds_raw_dim,
    //         _sk_hint,
    //     } = baked;

    //     let mut data_pd = padding::data(self, *padding_mode, cm.padding, *fft_size);

    //     let mut data_pd_fft = fft_processor.forward_with_scratch(&mut data_pd, scratch);
    //     let kernel_pd_fft = fft_processor.forward_with_scratch(kernel_pd, scratch);

    //     data_pd_fft.zip_mut_with(&kernel_pd_fft, |d, k| *d *= *k);
    //     // let mul_spec = data_pd_fft * kernel_pd_fft;

    //     let output = fft_processor.backward(data_pd_fft);

    //     output.slice_move(unsafe {
    //         SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
    //             start: kernel_raw_dim_with_dilation[i] as isize - 1,
    //             end: Some((pds_raw_dim[i]) as isize),
    //             step: cm.strides[i] as isize,
    //         }))
    //         .unwrap()
    //     })
    // }

    fn conv_fft(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, InElem>,
    ) -> Result<Array<InElem, Dim<[Ix; N]>>, crate::Error<N>> {
        let mut p = InElem::get_processor();
        self.conv_fft_with_processor(kernel, conv_mode, padding_mode, &mut p)
    }

    fn conv_fft_with_processor(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, InElem>,
        fft_processor: &mut impl Processor<T, InElem>,
    ) -> Result<Array<InElem, Dim<[Ix; N]>>, crate::Error<N>> {
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
            std::array::from_fn(|i| data_raw_dim[i] + cm.padding[i][0] + cm.padding[i][1]);
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

        let mut data_pd_fft = fft_processor.forward(&mut data_pd);
        let kernel_pd_fft = fft_processor.forward(&mut kernel_pd);

        data_pd_fft.zip_mut_with(&kernel_pd_fft, |d, k| *d *= *k);
        // let mul_spec = data_pd_fft * kernel_pd_fft;

        let output = fft_processor.backward(&mut data_pd_fft);

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
mod tests;
