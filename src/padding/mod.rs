use std::ops::Add;

use super::{BorderType, PaddingMode};

use ndarray::{
    Array, ArrayBase, Data, Dim, IntoDimension, Ix, OwnedRepr, RemoveAxis, SliceArg, SliceInfo,
    SliceInfoElem,
};
use num::traits::NumAssign;

mod dim;
mod half_dim;

pub type ExplicitPadding<const N: usize> = [[usize; 2]; N];

pub trait PaddingExt<const N: usize, T: num::traits::NumAssign + Copy, Output> {
    fn padding(&self, mode: PaddingMode<N, T>, padding_size: ExplicitPadding<N>) -> Output;
}

impl<const N: usize, T, S> PaddingExt<N, T, Array<T, Dim<[Ix; N]>>> for ArrayBase<S, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    fn padding(
        &self,
        mode: PaddingMode<N, T>,
        explicit_padding: ExplicitPadding<N>,
    ) -> Array<T, Dim<[Ix; N]>> {
        match mode {
            PaddingMode::Zeros => padding_const(self, explicit_padding, T::zero()),
            PaddingMode::Const(const_value) => padding_const(self, explicit_padding, const_value),
            PaddingMode::Replicate => padding_replicate(self, explicit_padding),
            PaddingMode::Reflect => padding_reflect(self, explicit_padding),
            PaddingMode::Circular => padding_circular(self, explicit_padding),
            PaddingMode::Custom(borders) => padding_custom(self, explicit_padding, borders),
            PaddingMode::Explicit(borders) => padding_explicit(self, explicit_padding, borders),
        }
    }
}

fn padding_const<const N: usize, T, S>(
    input: &ArrayBase<S, Dim<[Ix; N]>>,
    explicit_padding: ExplicitPadding<N>,
    const_value: T,
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    let output_dim = input.raw_dim().add(
        explicit_padding
            .map(|size| size[0] + size[1])
            .into_dimension(),
    );

    let mut output: Array<T, Dim<[Ix; N]>> = Array::from_elem(output_dim, const_value);

    let mut output_slice = output.slice_mut(unsafe {
        SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
            start: explicit_padding[i][0] as isize,
            end: Some((explicit_padding[i][0] + input.raw_dim()[i]) as isize),
            step: 1,
        }))
        .unwrap()
    });

    output_slice.assign(input);

    output
}

fn padding_replicate<const N: usize, T, S>(
    input: &ArrayBase<S, Dim<[Ix; N]>>,
    explicit_padding: ExplicitPadding<N>,
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    let mut output: ArrayBase<OwnedRepr<T>, Dim<[usize; N]>> =
        padding_const(input, explicit_padding, T::zero());

    explicit_padding
        .iter()
        .enumerate()
        .for_each(|(dim, &explicit_padding)| {
            dim::replicate(input.raw_dim(), &mut output, dim, explicit_padding);
        });

    output
}

fn padding_reflect<const N: usize, T, S>(
    input: &ArrayBase<S, Dim<[Ix; N]>>,
    explicit_padding: ExplicitPadding<N>,
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    let mut output: ArrayBase<OwnedRepr<T>, Dim<[usize; N]>> =
        padding_const(input, explicit_padding, T::zero());

    explicit_padding
        .iter()
        .enumerate()
        .for_each(|(dim, &explicit_padding)| {
            dim::reflect(input.raw_dim(), &mut output, dim, explicit_padding);
        });

    output
}

fn padding_circular<const N: usize, T, S>(
    input: &ArrayBase<S, Dim<[Ix; N]>>,
    explicit_padding: ExplicitPadding<N>,
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    let mut output: ArrayBase<OwnedRepr<T>, Dim<[usize; N]>> =
        padding_const(input, explicit_padding, T::zero());

    explicit_padding
        .iter()
        .enumerate()
        .for_each(|(dim, &explicit_padding)| {
            dim::circular(input.raw_dim(), &mut output, dim, explicit_padding);
        });

    output
}

fn padding_custom<const N: usize, T, S>(
    input: &ArrayBase<S, Dim<[Ix; N]>>,
    explicit_padding: ExplicitPadding<N>,
    borders: [BorderType<T>; N],
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    let mut output: ArrayBase<OwnedRepr<T>, Dim<[usize; N]>> =
        padding_const(input, explicit_padding, T::zero());

    explicit_padding
        .iter()
        .zip(borders.iter())
        .enumerate()
        .for_each(|(dim, (&explicit_padding, border))| match border {
            BorderType::Zeros => dim::constant(
                input.raw_dim(),
                &mut output,
                dim,
                explicit_padding,
                T::zero(),
            ),
            BorderType::Const(c) => {
                dim::constant(input.raw_dim(), &mut output, dim, explicit_padding, *c)
            }
            BorderType::Reflect => {
                dim::reflect(input.raw_dim(), &mut output, dim, explicit_padding)
            }
            BorderType::Replicate => {
                dim::replicate(input.raw_dim(), &mut output, dim, explicit_padding)
            }
            BorderType::Circular => {
                dim::circular(input.raw_dim(), &mut output, dim, explicit_padding)
            }
        });

    output
}

fn padding_explicit<const N: usize, T, S>(
    input: &ArrayBase<S, Dim<[Ix; N]>>,
    explicit_padding: ExplicitPadding<N>,
    borders: [[BorderType<T>; 2]; N],
) -> Array<T, Dim<[Ix; N]>>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    let mut output: ArrayBase<OwnedRepr<T>, Dim<[usize; N]>> =
        padding_const(input, explicit_padding, T::zero());

    explicit_padding
        .iter()
        .zip(borders.iter())
        .enumerate()
        .for_each(|(dim, (&explicit_padding, border))| {
            match border[0] {
                BorderType::Zeros => {
                    half_dim::constant_front(&mut output, dim, explicit_padding, T::zero())
                }
                BorderType::Const(c) => {
                    half_dim::constant_front(&mut output, dim, explicit_padding, c)
                }
                BorderType::Reflect => half_dim::reflect_front(&mut output, dim, explicit_padding),
                BorderType::Replicate => {
                    half_dim::replicate_front(&mut output, dim, explicit_padding)
                }
                BorderType::Circular => {
                    half_dim::circular_front(&mut output, dim, explicit_padding)
                }
            }
            match border[1] {
                BorderType::Zeros => half_dim::constant_back(
                    input.raw_dim(),
                    &mut output,
                    dim,
                    explicit_padding,
                    T::zero(),
                ),
                BorderType::Const(c) => {
                    half_dim::constant_back(input.raw_dim(), &mut output, dim, explicit_padding, c)
                }
                BorderType::Reflect => {
                    half_dim::reflect_back(input.raw_dim(), &mut output, dim, explicit_padding)
                }
                BorderType::Replicate => {
                    half_dim::replicate_back(input.raw_dim(), &mut output, dim, explicit_padding)
                }
                BorderType::Circular => {
                    half_dim::circular_back(input.raw_dim(), &mut output, dim, explicit_padding)
                }
            }
        });

    output
}

#[cfg(test)]
mod tests {
    use std::ops::AddAssign;

    use ndarray::prelude::*;

    use super::*;
    use crate::dilation::IntoKernelWithDilation;
    use crate::ConvMode;

    #[test]
    fn index_axis() {
        let mut arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        let mut sub = dbg!(arr.index_axis_mut(Axis(2), 0));

        sub += &array![[1, 1], [1, 1]];

        dbg!(&arr);

        assert_eq!(arr, array![[[2, 2], [4, 4]], [[6, 6], [8, 8]]]);
    }

    #[test]
    fn padding_replicate() {
        let arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        let kernel = array![
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        ];

        let explicit_conv = ConvMode::Full.unfold(&kernel.into_kernel_with_dilation());
        let explicit_padding = explicit_conv.padding;
        let arr_padded = arr.padding(PaddingMode::Replicate, explicit_padding);
        dbg!(arr_padded);

        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];

        let explicit_conv = ConvMode::Full.unfold(&kernel.into_kernel_with_dilation());
        let explicit_padding = explicit_conv.padding;
        let arr_padded = arr.padding(PaddingMode::Const(7), explicit_padding);
        dbg!(arr_padded);

        let arr = array![1, 2, 3];
        let kernel = array![1, 1, 1, 1];

        let explicit_conv = ConvMode::Same.unfold(&kernel.into_kernel_with_dilation());
        let explicit_padding = explicit_conv.padding;
        let arr_padded = arr.padding(PaddingMode::Zeros, explicit_padding);
        dbg!(arr_padded);
    }

    #[test]
    fn padding_custom() {
        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
        let kernel = kernel.into_kernel_with_dilation();

        let explicit_conv = ConvMode::Full.unfold(&kernel);
        let explicit_padding = explicit_conv.padding;
        let arr_padded = arr.padding(
            PaddingMode::Custom([BorderType::Replicate, BorderType::Circular]),
            explicit_padding,
        );
        assert_eq!(
            arr_padded,
            array![
                [1, 2, 1, 2, 1, 2],
                [1, 2, 1, 2, 1, 2],
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
                [3, 4, 3, 4, 3, 4],
                [3, 4, 3, 4, 3, 4]
            ]
        );

        let explicit_conv = ConvMode::Full.unfold(&kernel);
        let explicit_padding = explicit_conv.padding;
        let arr_padded = arr.padding(
            PaddingMode::Custom([BorderType::Reflect, BorderType::Const(7)]),
            explicit_padding,
        );
        assert_eq!(
            arr_padded,
            array![
                [7, 7, 0, 0, 7, 7],
                [7, 7, 3, 4, 7, 7],
                [7, 7, 1, 2, 7, 7],
                [7, 7, 3, 4, 7, 7],
                [7, 7, 1, 2, 7, 7],
                [7, 7, 3, 4, 7, 7]
            ]
        );

        dbg!(arr_padded);
    }

    #[test]
    fn tch_example() {
        let arr =
            tch::Tensor::from_slice2(&[[1, 2, 3], [3, 4, 5], [5, 6, 7]]).reshape([1, 1, 3, 3]);
        let arr_padded = arr
            .f_conv2d_padding(
                &tch::Tensor::from_slice2(&[[0, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape([1, 1, 3, 3]),
                Option::<tch::Tensor>::None,
                [1],
                "same",
                [1],
                1,
            )
            .unwrap();
        dbg!(&arr, &arr_padded);

        let arr = tch::Tensor::from_slice2(&[[1., 2.], [3., 4.]]).reshape([1, 1, 2, 2]);
        let arr_padded = arr.f_pad([1, 1, 1, 1], "circular", None).unwrap();

        arr.print();
        arr_padded.print();
    }

    #[test]
    fn aligned_with_libtorch() {
        let arr = array![[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]];
        let kernel = array![
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        ];
        let explicit_conv = ConvMode::Same.unfold(&kernel.into_kernel_with_dilation());
        let explicit_padding = explicit_conv.padding;
        check(&arr, PaddingMode::Zeros, explicit_padding);
        check(&arr, PaddingMode::Const(7), explicit_padding);
        check(&arr, PaddingMode::Replicate, explicit_padding);
        check(&arr, PaddingMode::Reflect, explicit_padding);
        check(&arr, PaddingMode::Circular, explicit_padding);

        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1, 1], [1, 1]];
        let explicit_conv = ConvMode::Full.unfold(&kernel.into_kernel_with_dilation());
        let explicit_padding = explicit_conv.padding;
        check(&arr, PaddingMode::Zeros, explicit_padding);
        check(&arr, PaddingMode::Const(7), explicit_padding);
        check(&arr, PaddingMode::Replicate, explicit_padding);
        check(&arr, PaddingMode::Reflect, explicit_padding);
        check(&arr, PaddingMode::Circular, explicit_padding);

        let arr = array![1, 2, 3];
        let kernel = array![1, 1, 1, 1];
        let explicit_conv = ConvMode::Same.unfold(&kernel.into_kernel_with_dilation());
        let explicit_padding = explicit_conv.padding;
        check(&arr, PaddingMode::Zeros, explicit_padding);
        check(&arr, PaddingMode::Const(7), explicit_padding);
        check(&arr, PaddingMode::Replicate, explicit_padding);
        check(&arr, PaddingMode::Reflect, explicit_padding);
        check(&arr, PaddingMode::Circular, explicit_padding);
    }

    fn check<T: AddAssign + Copy, const N: usize>(
        arr: &Array<T, Dim<[Ix; N]>>,
        padding_mode: PaddingMode<N, T>,
        explicit_padding: ExplicitPadding<N>,
    ) where
        T: num::traits::NumAssign + Copy + tch::kind::Element + std::fmt::Debug,
        Dim<[Ix; N]>: Dimension,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
        SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
        Dim<[Ix; N]>: RemoveAxis,
        f64: std::convert::From<T>,
        T: num::traits::FromPrimitive,
    {
        let ndarray_result = arr.padding(padding_mode, explicit_padding);
        dbg!(&ndarray_result);

        let shape = [1, 1]
            .iter()
            .chain(arr.shape())
            .map(|s| *s as i64)
            .collect::<Vec<_>>();
        let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
            .reshape(shape)
            .totype(tch::Kind::Float);

        let (mode, value) = match padding_mode {
            PaddingMode::Zeros => ("constant", Some(0.0)),
            PaddingMode::Const(c) => ("constant", Some(f64::from(c))),
            PaddingMode::Replicate => ("replicate", None),
            PaddingMode::Reflect => ("reflect", None),
            PaddingMode::Circular => ("circular", None),
            _ => unreachable!(),
        };

        let tensor_result = tensor
            .f_pad(
                explicit_padding
                    .into_iter()
                    .flatten()
                    .map(|p| p as i64)
                    .collect::<Vec<_>>(),
                mode,
                value,
            )
            .unwrap();

        dbg!(&tensor_result);
        tensor_result.print();

        assert_eq!(
            ndarray_result.into_raw_vec(),
            tensor_result
                .reshape(tensor_result.size().iter().product::<i64>())
                .iter::<f64>()
                .unwrap()
                .map(|v| T::from_f64(v).unwrap())
                .collect::<Vec<T>>()
        );
    }
}
