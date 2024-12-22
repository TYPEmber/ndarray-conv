//! Provides padding functions for FFT-based convolutions.
//!
//! These functions handle padding of input data and kernels to
//! appropriate sizes for efficient FFT calculations. Padding is
//! crucial for correctly implementing convolution via FFT.

use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, Data, Dim, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo, SliceInfoElem,
};
use num::traits::NumAssign;

use crate::{dilation::KernelWithDilation, padding::PaddingExt, ExplicitPadding, PaddingMode};

/// Pads the input data for FFT-based convolution.
///
/// This function takes the input data, padding mode, explicit padding, and desired FFT size
/// and returns a new array with the appropriate padding applied. The padding is applied
/// to each dimension according to the specified `padding_mode` and `explicit_padding`.
///
/// # Arguments
///
/// * `data`: The input data array.
/// * `padding_mode`: The padding mode to use (e.g., `PaddingMode::Zeros`, `PaddingMode::Reflect`).
/// * `explicit_padding`: An array specifying the padding for each dimension.
/// * `fft_size`: The desired size for FFT calculations. The output array will have these dimensions.
///
/// # Returns
///
/// A new array with the padded data, ready for FFT transformation.
pub fn data<T, S, const N: usize>(
    data: &ArrayBase<S, Dim<[Ix; N]>>,
    padding_mode: PaddingMode<N, T>,
    explicit_padding: ExplicitPadding<N>,
    fft_size: [usize; N],
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy + Debug,
    S: Data<Elem = T>,
    Dim<[Ix; N]>: RemoveAxis,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    // the key question is how to prove
    // <SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>> as SliceArg<Dim<[Ix; N]>>>::OutDim
    // is Dim<[Ix; N]>
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
{
    let mut buffer: Array<T, Dim<[Ix; N]>> = Array::from_elem(fft_size, T::zero());

    let raw_dim = data.raw_dim();
    let mut buffer_slice = buffer.slice_mut(unsafe {
        SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
            start: 0,
            end: Some((explicit_padding[i][0] + raw_dim[i] + explicit_padding[i][1]) as isize),
            step: 1,
        }))
        .unwrap()
    });

    data.padding_in(&mut buffer_slice, padding_mode, explicit_padding);

    buffer
}

/// Pads the kernel for FFT-based convolution.
///
/// This function takes the kernel, expands it with dilations, and pads it with zeros to the
/// desired FFT size, preparing it for FFT transformation. The kernel is also reversed
/// in each dimension as required for convolution via FFT.
///
/// # Arguments
///
/// * `kwd`: The kernel with dilation information.
/// * `fft_size`: The desired size for FFT calculations. The output array will have these dimensions.
///
/// # Returns
///
/// A new array with the padded and reversed kernel, ready for FFT transformation.
pub fn kernel<'a, T, S, const N: usize>(
    kwd: KernelWithDilation<'a, S, N>,
    fft_size: [usize; N],
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy + Debug + 'a,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
{
    let mut buffer: Array<T, Dim<[Ix; N]>> = Array::from_elem(fft_size, T::zero());

    let kernel = kwd.kernel;

    let kernel_raw_dim = kernel.raw_dim();
    let kernel_raw_dim_with_dilation: [usize; N] =
        std::array::from_fn(|i| kernel_raw_dim[i] * kwd.dilation[i] - kwd.dilation[i] + 1);

    let mut buffer_slice = buffer.slice_mut(unsafe {
        SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
            start: 0,
            end: Some(kernel_raw_dim_with_dilation[i] as isize),
            // use negative stride to make kernel reverse
            step: -(kwd.dilation[i] as isize),
        }))
        .unwrap()
    });

    buffer_slice.zip_mut_with(kernel, |b, &k| *b = k);

    buffer
}

#[cfg(test)]
mod tests {
    use crate::{
        dilation::{IntoKernelWithDilation, WithDilation},
        BorderType, ConvMode,
    };
    use ndarray::prelude::*;

    use super::*;

    #[test]
    fn data_padding() {
        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
        let kernel = kernel.into_kernel_with_dilation();

        let explicit_conv = ConvMode::Full.unfold(&kernel);
        let explicit_padding = explicit_conv.padding;

        let arr_padded = data(
            &arr,
            PaddingMode::Custom([BorderType::Const(7), BorderType::Const(8)]),
            // PaddingMode::Const(7),
            explicit_padding,
            [8, 8],
        );

        assert_eq!(
            arr_padded,
            array![
                [8, 8, 7, 7, 8, 8, 0, 0],
                [8, 8, 7, 7, 8, 8, 0, 0],
                [8, 8, 1, 2, 8, 8, 0, 0],
                [8, 8, 3, 4, 8, 8, 0, 0],
                [8, 8, 7, 7, 8, 8, 0, 0],
                [8, 8, 7, 7, 8, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn kernel_padding() {
        let _arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let kernel = kernel.with_dilation([2, 3]).into_kernel_with_dilation();

        let explicit_conv = ConvMode::Full.unfold(&kernel);
        let _explicit_padding = explicit_conv.padding;

        let kernel_padded = super::kernel(kernel, [8, 8]);

        dbg!(&kernel_padded);

        assert_eq!(
            kernel_padded,
            array![
                [9, 0, 0, 8, 0, 0, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [6, 0, 0, 5, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 2, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }
}
