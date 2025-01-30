//! Provides convolution operations for `ndarray` arrays.
//! Includes standard convolution and related utilities.

use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, ArrayView, Data, Dim, Dimension, IntoDimension, Ix, RawData, RemoveAxis,
    SliceArg, SliceInfo, SliceInfoElem,
};
use num::traits::NumAssign;

use crate::{
    dilation::{IntoKernelWithDilation, KernelWithDilation},
    padding::PaddingExt,
    ConvMode, PaddingMode,
};

#[cfg(test)]
mod tests;

/// Represents explicit convolution parameters after unfolding from `ConvMode`.
///
/// This struct holds padding and strides information used directly
/// by the convolution algorithm.
pub struct ExplicitConv<const N: usize> {
    pub padding: [[usize; 2]; N],
    pub strides: [usize; N],
}

impl<const N: usize> ConvMode<N> {
    pub(crate) fn unfold<S>(self, kernel: &KernelWithDilation<S, N>) -> ExplicitConv<N>
    where
        S: ndarray::RawData,
        Dim<[Ix; N]>: Dimension,
    {
        let kernel_dim = kernel.kernel.raw_dim();
        let kernel_dim: [usize; N] = std::array::from_fn(|i|
                // k + (k - 1) * (d - 1)
                kernel_dim[i] * kernel.dilation[i] - kernel.dilation[i] + 1);

        match self {
            ConvMode::Full => ExplicitConv {
                padding: std::array::from_fn(|i| [kernel_dim[i] - 1; 2]),
                strides: [1; N],
            },
            ConvMode::Same => ExplicitConv {
                padding: std::array::from_fn(|i| {
                    let k_size = kernel_dim[i];
                    if k_size % 2 == 0 {
                        [(k_size - 1) / 2 + 1, (k_size - 1) / 2]
                    } else {
                        [(k_size - 1) / 2; 2]
                    }
                }),
                strides: [1; N],
            },
            ConvMode::Valid => ExplicitConv {
                padding: [[0; 2]; N],
                strides: [1; N],
            },
            ConvMode::Custom { padding, strides } => ExplicitConv {
                padding: padding.map(|pad| [pad; 2]),
                strides,
            },
            ConvMode::Explicit { padding, strides } => ExplicitConv { padding, strides },
        }
    }
}

/// Extends `ndarray`'s `ArrayBase` with convolution operations.
///
/// This trait adds the `conv` method to `ArrayBase`, enabling
/// standard convolution operations on N-dimensional arrays.
///
/// # Type Parameters
///
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `SK`: The data storage type of the kernel array.
pub trait ConvExt<'a, T, S, SK, const N: usize>
where
    T: NumAssign + Copy,
    S: RawData,
    SK: RawData,
{
    /// Performs a standard convolution operation.
    ///
    /// This method convolves the input array with a given kernel,
    /// using the specified convolution mode and padding.
    ///
    /// # Arguments
    ///
    /// *   `kernel`: The convolution kernel.
    /// *   `conv_mode`: The convolution mode (`Full`, `Same`, `Valid`, `Custom`, `Explicit`).
    /// *   `padding_mode`: The padding mode (`Zeros`, `Const`, `Reflect`, `Replicate`, `Circular`, `Custom`, `Explicit`).
    ///
    /// # Returns
    ///
    fn conv(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Result<Array<T, Dim<[Ix; N]>>, crate::Error<N>>;
}

impl<'a, T, S, SK, const N: usize> ConvExt<'a, T, S, SK, N> for ArrayBase<S, Dim<[Ix; N]>>
where
    T: NumAssign + Copy + Debug,
    S: Data<Elem = T> + 'a,
    SK: Data<Elem = T> + 'a,
    Dim<[Ix; N]>: RemoveAxis,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
{
    fn conv(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Result<Array<T, Dim<[Ix; N]>>, crate::Error<N>> {
        let kwd = kernel.into_kernel_with_dilation();

        let self_raw_dim = self.raw_dim();
        if self.shape().iter().product::<usize>() == 0 {
            return Err(crate::Error::DataShape(self_raw_dim));
        }

        let kernel_raw_dim = kwd.kernel.raw_dim();
        if kwd.kernel.shape().iter().product::<usize>() == 0 {
            return Err(crate::Error::DataShape(kernel_raw_dim));
        }

        let kernel_raw_dim_with_dilation: [usize; N] =
            std::array::from_fn(|i| kernel_raw_dim[i] * kwd.dilation[i] - kwd.dilation[i] + 1);

        let cm = conv_mode.unfold(&kwd);
        let pds = self.padding(padding_mode, cm.padding);

        let pds_raw_dim = pds.raw_dim();
        if !(0..N).all(|i| kernel_raw_dim_with_dilation[i] <= pds_raw_dim[i]) {
            return Err(crate::Error::MismatchShape(
                conv_mode,
                kernel_raw_dim_with_dilation,
            ));
        }

        let offset_list = kwd.gen_offset_list(pds.strides());

        let output_shape: [usize; N] = std::array::from_fn(|i| {
            (cm.padding[i][0] + cm.padding[i][1] + self_raw_dim[i]
                - kernel_raw_dim_with_dilation[i])
                / cm.strides[i]
                + 1
        });
        let mut ret = Array::zeros(output_shape);

        let shape: [usize; N] = std::array::from_fn(|i| ret.raw_dim()[i]);
        let strides: [usize; N] =
            std::array::from_fn(|i| cm.strides[i] * pds.strides()[i] as usize);

        // dbg!(&offset_list);
        // dbg!(strides);

        unsafe {
            // use raw pointer to improve performance.
            let p: *mut T = ret.as_mut_ptr();

            // use ArrayView's iter without handle strides
            let view = ArrayView::from_shape(
                ndarray::ShapeBuilder::strides(shape, strides),
                pds.as_slice().unwrap(),
            )
            .unwrap();

            view.iter().enumerate().for_each(|(i, cur)| {
                let mut tmp_res = T::zero();

                offset_list.iter().for_each(|(tmp_offset, tmp_kernel)| {
                    tmp_res += *(cur as *const T).offset(*tmp_offset) * *tmp_kernel
                });

                *p.add(i) = tmp_res;
            });
        }

        Ok(ret)
    }
}
