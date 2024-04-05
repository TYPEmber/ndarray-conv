use std::fmt::Debug;

use ndarray::{
    Array, ArrayBase, Data, Dim, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo, SliceInfoElem,
};
use num::traits::NumAssign;

use crate::{padding::PaddingExt, ExplicitPadding, PaddingMode};

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

pub fn kernel<'a, T, S, D, const N: usize>(
    kernel: &'a ArrayBase<S, D>,
    fft_size: [usize; N],
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy + Debug + 'a,
    S: Data<Elem = T>,
    D: RemoveAxis,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
{
    let mut buffer: Array<T, Dim<[Ix; N]>> = Array::from_elem(fft_size, T::zero());

    let raw_dim = kernel.raw_dim();
    let mut buffer_slice = buffer.slice_mut(unsafe {
        SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
            start: 0,
            end: Some(raw_dim[i] as isize),
            step: 1,
        }))
        .unwrap()
    });

    buffer_slice
        .iter_mut()
        .zip(
            kernel
                .as_slice()
                .unwrap_or(
                    Array::from_shape_vec(kernel.raw_dim(), kernel.iter().cloned().collect())
                        .unwrap()
                        .as_slice()
                        .unwrap(),
                )
                .iter()
                .rev(),
        )
        .for_each(|(b, &k)| *b = k);

    buffer
}

#[cfg(test)]
mod tests {
    use crate::{dilation::IntoKernelWithDilation, BorderType, ConvMode};
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
        let kernel = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
        let kernel = kernel.into_kernel_with_dilation();

        let explicit_conv = ConvMode::Full.unfold(&kernel);
        let _explicit_padding = explicit_conv.padding;

        let kernel_padded = super::kernel(kernel.kernel, [8, 8]);

        assert_eq!(
            kernel_padded,
            array![
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }
}
