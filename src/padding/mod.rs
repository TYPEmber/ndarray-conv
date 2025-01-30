//! Provides padding functionality for ndarray arrays.
//!
//! This module defines the `PaddingExt` trait, which extends the `ArrayBase`
//! struct from the `ndarray` crate with methods for padding arrays using
//! different padding modes. It also provides helper functions for
//! applying specific types of padding.

use super::{BorderType, PaddingMode};

use ndarray::{
    Array, ArrayBase, Data, DataMut, Dim, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo,
    SliceInfoElem,
};
use num::traits::NumAssign;

pub(crate) mod dim;
mod half_dim;

/// Represents explicit padding sizes for each dimension.
pub type ExplicitPadding<const N: usize> = [[usize; 2]; N];

/// Extends `ndarray`'s `ArrayBase` with padding operations.
///
/// This trait provides the `padding` and `padding_in` methods for adding
/// padding to an array using various modes, like zero padding, constant
/// padding, replication, reflection, and circular padding.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `Output`: The type of the padded array returned by `padding`, typically an `Array<T, Dim<[Ix; N]>>`.
pub trait PaddingExt<const N: usize, T: num::traits::NumAssign + Copy, Output> {
    /// Returns a new array with the specified padding applied.
    ///
    /// This method creates a new array with the dimensions and padding specified by
    /// `mode` and `padding_size`. It calls the `padding_in` method internally to handle the padding itself.
    ///
    /// # Arguments
    ///
    /// * `mode`: The padding mode (`Zeros`, `Const`, `Reflect`, `Replicate`, `Circular`, `Custom`, `Explicit`).
    /// * `padding_size`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
    ///
    /// # Returns
    /// A new `Array` with the padded data.
    fn padding(&self, mode: PaddingMode<N, T>, padding_size: ExplicitPadding<N>) -> Output;

    /// Modifies the buffer in-place by applying padding using the specified mode.
    ///
    /// This method directly modifies the provided buffer by adding padding to its content.
    ///
    /// # Type Parameters
    ///
    /// *   `SO`: The data storage type of the output buffer.
    /// *   `DO`: The dimension type of the output buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer`: A mutable reference to the array to be padded.
    /// * `mode`: The padding mode (`Zeros`, `Const`, `Reflect`, `Replicate`, `Circular`, `Custom`, `Explicit`).
    /// * `padding_size`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
    fn padding_in<SO: DataMut<Elem = T>, DO: RemoveAxis>(
        &self,
        buffer: &mut ArrayBase<SO, DO>,
        mode: PaddingMode<N, T>,
        padding_size: ExplicitPadding<N>,
    ) where
        T: NumAssign + Copy,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
        SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
        Dim<[Ix; N]>: RemoveAxis,
        SliceInfo<[SliceInfoElem; N], DO, DO>: SliceArg<DO>;
}

impl<const N: usize, T, S, D> PaddingExt<N, T, Array<T, Dim<[Ix; N]>>> for ArrayBase<S, D>
where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis + IntoDimension,
{
    fn padding(
        &self,
        mode: PaddingMode<N, T>,
        explicit_padding: ExplicitPadding<N>,
    ) -> Array<T, Dim<[Ix; N]>> {
        let c = match mode {
            PaddingMode::Const(c) => c,
            _ => T::zero(),
        };

        let raw_dim = self.raw_dim();

        let output_dim =
            std::array::from_fn(|i| raw_dim[i] + explicit_padding[i][0] + explicit_padding[i][1]);

        let mut output: Array<T, Dim<[Ix; N]>> = Array::from_elem(output_dim, c);

        padding_const(self, &mut output, explicit_padding);

        match mode {
            PaddingMode::Replicate => padding_replicate(self, &mut output, explicit_padding),
            PaddingMode::Reflect => padding_reflect(self, &mut output, explicit_padding),
            PaddingMode::Circular => padding_circular(self, &mut output, explicit_padding),
            PaddingMode::Custom(borders) => {
                padding_custom(self, &mut output, explicit_padding, borders)
            }
            PaddingMode::Explicit(borders) => {
                padding_explicit(self, &mut output, explicit_padding, borders)
            }
            _ => {}
        };

        output
    }

    fn padding_in<SO, DO>(
        &self,
        buffer: &mut ArrayBase<SO, DO>,
        mode: PaddingMode<N, T>,
        explicit_padding: ExplicitPadding<N>,
    ) where
        T: NumAssign + Copy,
        S: Data<Elem = T>,
        SO: DataMut<Elem = T>,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
        SliceInfo<[SliceInfoElem; N], DO, DO>: SliceArg<DO>,
        Dim<[Ix; N]>: RemoveAxis,
        DO: RemoveAxis,
    {
        padding_const(self, buffer, explicit_padding);

        match mode {
            PaddingMode::Const(c) => {
                explicit_padding
                    .iter()
                    .enumerate()
                    .for_each(|(dim, &explicit_padding)| {
                        dim::constant(self.raw_dim(), buffer, dim, explicit_padding, c);
                    })
            }
            PaddingMode::Replicate => padding_replicate(self, buffer, explicit_padding),
            PaddingMode::Reflect => padding_reflect(self, buffer, explicit_padding),
            PaddingMode::Circular => padding_circular(self, buffer, explicit_padding),
            PaddingMode::Custom(borders) => padding_custom(self, buffer, explicit_padding, borders),
            PaddingMode::Explicit(borders) => {
                padding_explicit(self, buffer, explicit_padding, borders)
            }
            _ => {}
        };
    }
}

/// Applies padding using a constant value to the specified slice of the output array.
///
/// This function copies the input array to a specific slice of the output array, leaving the rest of the
/// output array with the default padding value, which is typically a zero or a constant, depending on the padding mode.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `D`: The dimension type of the input array.
/// *   `SO`: The data storage type of the output array.
/// *   `DO`: The dimension type of the output array.
///
/// # Arguments
///
/// * `input`: The input array to pad.
/// * `output`: A mutable reference to the array where the padded result will be stored.
/// * `explicit_padding`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
pub(crate) fn padding_const<const N: usize, T, S, D, SO, DO>(
    input: &ArrayBase<S, D>,
    output: &mut ArrayBase<SO, DO>,
    explicit_padding: ExplicitPadding<N>,
) where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    SO: DataMut<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    // SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], DO, DO>: SliceArg<DO>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis,
    DO: RemoveAxis,
{
    let mut output_slice = output.slice_mut(unsafe {
        SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
            start: explicit_padding[i][0] as isize,
            end: Some((explicit_padding[i][0] + input.raw_dim()[i]) as isize),
            step: 1,
        }))
        .unwrap()
    });

    output_slice.assign(input);
}

/// Applies replicate padding to the specified slice of the output array.
///
/// This function uses the `dim::replicate` function to add replicate padding
/// to each dimension of the output array.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `D`: The dimension type of the input array.
/// *   `SO`: The data storage type of the output array.
/// *   `DO`: The dimension type of the output array.
///
/// # Arguments
///
/// * `input`: The input array to pad.
/// * `output`: A mutable reference to the array where the padded result will be stored.
/// * `explicit_padding`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
fn padding_replicate<const N: usize, T, S, D, SO, DO>(
    input: &ArrayBase<S, D>,
    output: &mut ArrayBase<SO, DO>,
    explicit_padding: ExplicitPadding<N>,
) where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    SO: DataMut<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis + IntoDimension,
    DO: RemoveAxis,
{
    explicit_padding
        .iter()
        .enumerate()
        .for_each(|(dim, &explicit_padding)| {
            dim::replicate(input.raw_dim(), output, dim, explicit_padding);
        });
}

/// Applies reflect padding to the specified slice of the output array.
///
/// This function uses the `dim::reflect` function to add reflect padding
/// to each dimension of the output array.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `D`: The dimension type of the input array.
/// *   `SO`: The data storage type of the output array.
/// *   `DO`: The dimension type of the output array.
///
/// # Arguments
///
/// * `input`: The input array to pad.
/// * `output`: A mutable reference to the array where the padded result will be stored.
/// * `explicit_padding`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
fn padding_reflect<const N: usize, T, S, D, SO, DO>(
    input: &ArrayBase<S, D>,
    output: &mut ArrayBase<SO, DO>,
    explicit_padding: ExplicitPadding<N>,
) where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    SO: DataMut<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis,
    DO: RemoveAxis,
{
    explicit_padding
        .iter()
        .enumerate()
        .for_each(|(dim, &explicit_padding)| {
            dim::reflect(input.raw_dim(), output, dim, explicit_padding);
        });
}

/// Applies circular padding to the specified slice of the output array.
///
/// This function uses the `dim::circular` function to add circular padding
/// to each dimension of the output array.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `D`: The dimension type of the input array.
/// *   `SO`: The data storage type of the output array.
/// *   `DO`: The dimension type of the output array.
///
/// # Arguments
///
/// * `input`: The input array to pad.
/// * `output`: A mutable reference to the array where the padded result will be stored.
/// * `explicit_padding`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
fn padding_circular<const N: usize, T, S, D, SO, DO>(
    input: &ArrayBase<S, D>,
    output: &mut ArrayBase<SO, DO>,
    explicit_padding: ExplicitPadding<N>,
) where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    SO: DataMut<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis,
    DO: RemoveAxis,
{
    explicit_padding
        .iter()
        .enumerate()
        .for_each(|(dim, &explicit_padding)| {
            dim::circular(input.raw_dim(), output, dim, explicit_padding);
        });
}

/// Applies custom padding to the specified slice of the output array using `BorderType` for each dimension.
///
/// This function uses the `dim::constant`, `dim::reflect`, `dim::replicate`,
/// or `dim::circular` function based on the corresponding `BorderType` specified in the `borders` argument,
/// to add padding to each dimension of the output array.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `D`: The dimension type of the input array.
/// *   `SO`: The data storage type of the output array.
/// *   `DO`: The dimension type of the output array.
///
/// # Arguments
///
/// * `input`: The input array to pad.
/// * `output`: A mutable reference to the array where the padded result will be stored.
/// * `explicit_padding`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
/// * `borders`: An array containing a `BorderType` enum for each dimension.
fn padding_custom<const N: usize, T, S, D, SO, DO>(
    input: &ArrayBase<S, D>,
    output: &mut ArrayBase<SO, DO>,
    explicit_padding: ExplicitPadding<N>,
    borders: [BorderType<T>; N],
) where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    SO: DataMut<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis,
    DO: RemoveAxis,
{
    explicit_padding
        .iter()
        .zip(borders.iter())
        .enumerate()
        .for_each(|(dim, (&explicit_padding, border))| match border {
            BorderType::Zeros => {
                dim::constant(input.raw_dim(), output, dim, explicit_padding, T::zero())
            }
            BorderType::Const(c) => {
                dim::constant(input.raw_dim(), output, dim, explicit_padding, *c)
            }
            BorderType::Reflect => dim::reflect(input.raw_dim(), output, dim, explicit_padding),
            BorderType::Replicate => dim::replicate(input.raw_dim(), output, dim, explicit_padding),
            BorderType::Circular => dim::circular(input.raw_dim(), output, dim, explicit_padding),
        });
}

/// Applies explicit padding to the specified slice of the output array using `BorderType` for each side of each dimension.
///
/// This function uses the `half_dim::constant_front`, `half_dim::constant_back`,
/// `half_dim::reflect_front`, `half_dim::reflect_back`, `half_dim::replicate_front`,
/// `half_dim::replicate_back`, `half_dim::circular_front`, and `half_dim::circular_back`
/// functions based on the corresponding `BorderType` specified in the `borders` argument,
/// to add padding to each dimension of the output array.
///
/// # Type Parameters
///
/// *   `N`: The number of dimensions of the array.
/// *   `T`: The numeric type of the array elements.
/// *   `S`: The data storage type of the input array.
/// *   `D`: The dimension type of the input array.
/// *   `SO`: The data storage type of the output array.
/// *   `DO`: The dimension type of the output array.
///
/// # Arguments
///
/// * `input`: The input array to pad.
/// * `output`: A mutable reference to the array where the padded result will be stored.
/// * `explicit_padding`: An array representing the padding sizes for each dimension in the form `[[front, back]; N]`.
/// * `borders`: An array containing an array of two `BorderType` enums for each dimension.
fn padding_explicit<const N: usize, T, S, D, SO, DO>(
    input: &ArrayBase<S, D>,
    output: &mut ArrayBase<SO, DO>,
    explicit_padding: ExplicitPadding<N>,
    borders: [[BorderType<T>; 2]; N],
) where
    T: NumAssign + Copy,
    S: Data<Elem = T>,
    SO: DataMut<Elem = T>,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    D: RemoveAxis,
    DO: RemoveAxis,
{
    explicit_padding
        .iter()
        .zip(borders.iter())
        .enumerate()
        .for_each(|(dim, (&explicit_padding, border))| {
            match border[0] {
                BorderType::Zeros => {
                    half_dim::constant_front(output, dim, explicit_padding, T::zero())
                }
                BorderType::Const(c) => half_dim::constant_front(output, dim, explicit_padding, c),
                BorderType::Reflect => half_dim::reflect_front(output, dim, explicit_padding),
                BorderType::Replicate => half_dim::replicate_front(output, dim, explicit_padding),
                BorderType::Circular => half_dim::circular_front(output, dim, explicit_padding),
            }
            match border[1] {
                BorderType::Zeros => half_dim::constant_back(
                    input.raw_dim(),
                    output,
                    dim,
                    explicit_padding,
                    T::zero(),
                ),
                BorderType::Const(c) => {
                    half_dim::constant_back(input.raw_dim(), output, dim, explicit_padding, c)
                }
                BorderType::Reflect => {
                    half_dim::reflect_back(input.raw_dim(), output, dim, explicit_padding)
                }
                BorderType::Replicate => {
                    half_dim::replicate_back(input.raw_dim(), output, dim, explicit_padding)
                }
                BorderType::Circular => {
                    half_dim::circular_back(input.raw_dim(), output, dim, explicit_padding)
                }
            }
        });
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

    // #[test]
    // fn padding_replicate() {
    //     let arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    //     let kernel = array![
    //         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    //         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    //         [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    //     ];

    //     let explicit_conv = ConvMode::Full.unfold(&kernel.into_kernel_with_dilation());
    //     let explicit_padding = explicit_conv.padding;
    //     let arr_padded = arr.padding(PaddingMode::Replicate, explicit_padding);
    //     dbg!(arr_padded);

    //     let arr = array![[1, 2], [3, 4]];
    //     let kernel = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];

    //     let explicit_conv = ConvMode::Full.unfold(&kernel.into_kernel_with_dilation());
    //     let explicit_padding = explicit_conv.padding;
    //     let arr_padded = arr.padding(PaddingMode::Const(7), explicit_padding);
    //     dbg!(arr_padded);

    //     let arr = array![1, 2, 3];
    //     let kernel = array![1, 1, 1, 1];

    //     let explicit_conv = ConvMode::Same.unfold(&kernel.into_kernel_with_dilation());
    //     let explicit_padding = explicit_conv.padding;
    //     let arr_padded = arr.padding(PaddingMode::Zeros, explicit_padding);
    //     dbg!(arr_padded);
    // }

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
