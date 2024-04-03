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

pub struct ExplicitConv<const N: usize> {
    pub padding: [[usize; 2]; N],
    pub strides: [usize; N],
}

impl<const N: usize> ConvMode<N> {
    pub(crate) fn unfold<S>(self, kernel: &KernelWithDilation<S, N>) -> ExplicitConv<N>
    where
        S: ndarray::RawData,
        Dim<[Ix; N]>: Dimension,
        // [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
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

pub trait ConvExt<'a, T, TK, S, SK, const N: usize>
where
    T: NumAssign + Copy,
    TK: NumAssign + Copy,
    S: RawData,
    SK: RawData,
{
    fn conv(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>>;
}

impl<'a, T, TK, S, SK, const N: usize> ConvExt<'a, T, TK, S, SK, N> for ArrayBase<S, Dim<[Ix; N]>>
where
    T: NumAssign + Copy + Debug,
    TK: NumAssign + Copy + Debug,
    S: Data<Elem = T> + 'a,
    SK: Data<Elem = TK> + 'a,
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    T: From<TK>,
{
    fn conv(
        &self,
        kernel: impl IntoKernelWithDilation<'a, SK, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>> {
        let kernel = kernel.into_kernel_with_dilation();

        // if kernel.dilation.iter == 0 {
        //     return None;
        // }

        let cm = conv_mode.unfold(&kernel);
        let pds = self.padding(padding_mode, cm.padding);

        let offset_list = kernel.gen_offset_list(pds.strides());

        let self_raw_dim = self.raw_dim();
        let kernel_raw_dim = kernel.kernel.raw_dim();
        let kernel_raw_dim_with_dilation: [usize; N] = std::array::from_fn(|i| {
            kernel_raw_dim[i] * kernel.dilation[i] - kernel.dilation[i] + 1
        });
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
                    tmp_res += *(cur as *const T).offset(*tmp_offset) * T::from(*tmp_kernel)
                });

                *p.add(i) = tmp_res;
            });
        }

        Some(ret)
    }
}
