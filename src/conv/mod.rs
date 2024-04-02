use std::fmt::Debug;

use ndarray::{
    prelude::*, Data, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo, SliceInfoElem,
};
use num::traits::NumAssign;

use crate::{
    dilation::{IntoKernelWithDilation, KernelWithDilation},
    padding::PaddingExt,
    ConvMode, ExplicitPadding, PaddingMode,
};

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

pub trait ConvExt<T: NumAssign + Copy, S: ndarray::RawData, const N: usize> {
    fn conv(
        &self,
        kernel: impl IntoKernelWithDilation<S, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>>;
}

impl<T: NumAssign + Copy, S: ndarray::RawData, const N: usize> ConvExt<T, S, N>
    for ArrayBase<S, Dim<[Ix; N]>>
where
    T: num::traits::NumAssign + Copy + Debug,
    S: Data<Elem = T>,
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    fn conv(
        &self,
        kernel: impl IntoKernelWithDilation<S, N>,
        conv_mode: ConvMode<N>,
        padding_mode: PaddingMode<N, T>,
    ) -> Option<Array<T, Dim<[Ix; N]>>> {
        let kernel = kernel.into_kernel_with_dilation();

        let cm = conv_mode.unfold(&kernel);
        let pds = self.padding(padding_mode, cm.padding);

        let offset_list = kernel.gen_offset_list(pds.shape());

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

        // dbg!(ret.shape());

        let shape: [usize; N] = std::array::from_fn(|i| ret.raw_dim()[i]);
        let strides: [usize; N] =
            std::array::from_fn(|i| cm.strides[i] * pds.strides()[i] as usize);

        // dbg!(&pds);
        // dbg!(shape.strides(strides));

        unsafe {
            let view =
                ArrayView::from_shape(shape.strides(strides), pds.as_slice().unwrap()).unwrap();

            view.iter().zip(ret.iter_mut()).for_each(|(cur, r)| {
                let mut tmp_res = T::zero();

                offset_list.iter().for_each(|(tmp_offset, tmp_kernel)| {
                    tmp_res += *(cur as *const T).offset(*tmp_offset) * *tmp_kernel
                });

                *r = tmp_res;
            });
        }

        Some(ret)
    }
}

#[cfg(test)]
mod tests {
    use crate::dilation::WithDilation;

    use super::*;

    #[test]
    fn tch_conv2d() {
        // let a = vec![1, 2, 3];
        // let b = a
        //     .iter()
        //     .flat_map(|&v| std::iter::repeat(v).take(4))
        //     .collect::<Vec<_>>();
        let tensor = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            .to_dtype(tch::Kind::Float, false, true)
            .reshape([1, 1, 3, 3]);
        let kernel = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            .to_dtype(tch::Kind::Float, false, true)
            .reshape([1, 1, 3, 3]);
        // let kernel = tch::Tensor::from_slice2(&[[1]])
        //     .to_dtype(tch::Kind::Float, false, true)
        //     .reshape([1, 1, 1, 1]);
        // let result = tensor.f_conv2d::<tch::Tensor>(&kernel, None, 1, [1, 1], 3i64, 1);
        let result = tensor.f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1);
        result.unwrap().print();
    }

    #[test]
    fn test_conv() {
        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1], [1, 1]];

        let res = arr
            .conv(
                kernel,
                ConvMode::Custom {
                    padding: [1, 2],
                    strides: [2, 2],
                },
                PaddingMode::Zeros,
            )
            .unwrap();
        assert_eq!(res, array![[0, 3, 0], [0, 7, 0]]);
        dbg!(res);

        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1], [1, 1]];

        let res = arr
            .conv(kernel, ConvMode::Full, PaddingMode::Zeros)
            .unwrap();
        assert_eq!(res, array![[1, 3, 2], [4, 10, 6], [3, 7, 4]]);
        dbg!(res);

        let arr = array![1, 2, 3, 4, 5, 6];
        let kernel = array![1, 1, 1];

        let res = arr
            .conv(
                kernel,
                ConvMode::Custom {
                    padding: [4],
                    strides: [2],
                },
                PaddingMode::Zeros,
            )
            .unwrap();
        assert_eq!(res, array![0, 1, 6, 12, 11, 0]);
        dbg!(res);


        let arr = array![1, 2, 3, 4, 5, 6];
        let kernel = array![1, 1, 1];

        let res = arr
            .conv(
                kernel.with_dilation(2),
                ConvMode::Custom {
                    padding: [4],
                    strides: [2],
                },
                PaddingMode::Zeros,
            )
            .unwrap();
        assert_eq!(res, array![1, 4, 9, 8, 5]);
        dbg!(res);
    }
}
