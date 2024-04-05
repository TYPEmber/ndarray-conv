use std::fmt::Debug;

use ndarray::{
    s, Array, ArrayBase, ArrayView, Data, Dim, Dimension, IntoDimension, Ix, RawData, RemoveAxis,
    SliceArg, SliceInfo, SliceInfoElem,
};
use num::traits::NumAssign;

use crate::{
    conv::ExplicitConv,
    dilation::{self, IntoKernelWithDilation, KernelWithDilation},
    padding::{self, padding_const, PaddingExt},
    ConvExt, ExplicitPadding, PaddingMode,
};

// trait TTT<D: Dimension>: SliceArg<D, OutDim = D> {
//     type OutDim;
// }

// impl<T, const N: usize> TTT<Dim<[Ix; N]>> for SliceInfo<T, Dim<[Ix; N]>, Dim<[Ix; N]>>
// where
//     Dim<[Ix; N]>: RemoveAxis,
//     SliceInfo<T, Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
// {
//     type OutDim = Dim<[Ix; N]>;
// }

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
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
    // SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    // SliceInfo<[SliceInfoElem; N], <SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>,
    //     Dim<[Ix; N]>> as SliceArg<Dim<[Ix; N]>>>::OutDim, <SliceInfo<[SliceInfoElem; N],
    //     Dim<[Ix; N]>, Dim<[Ix; N]>> as SliceArg<Dim<[Ix; N]>>>::OutDim>:
    //     SliceArg<<SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>> as SliceArg<Dim<[Ix; N]>>>::OutDim>,
    // <SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>> as TTT<Dim<[Ix; N]>>>::OutDim:
    //     RemoveAxis,
    // the key question is how to prove
    // <SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>> as SliceArg<Dim<[Ix; N]>>>::OutDim
    // is Dim<[Ix; N]>
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

pub fn kernel<'a, T, S, D, const N: usize>(kernel: &'a ArrayBase<S, D>, fft_size: [usize; N])
where
    T: NumAssign + Copy + Debug + 'a,
    S: Data<Elem = T>,
    D: RemoveAxis,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    // ndarray::iter::Iter<'a, T, D>: std::iter::DoubleEndedIterator<Item = &'a T>,
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
        .zip(kernel.as_slice().unwrap().iter().rev())
        .for_each(|(b, &k)| *b = k);
}

#[cfg(test)]
mod tests {
    use crate::{dilation::IntoKernelWithDilation, BorderType, ConvMode};
    use ndarray::prelude::*;

    use super::*;

    #[test]
    fn t() {
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

        dbg!(arr_padded);
    }
}
