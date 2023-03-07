use std::{marker::PhantomData, ops::Index, slice::SliceIndex};

use crate::*;

#[derive(Debug)]
pub struct ExplicitConv<const N: usize> {
    pub pad: [[usize; 2]; N],
    pub stride: [usize; N],
}

#[derive(Debug)]
pub struct ExplictPadding<const N: usize, T: num::traits::NumAssign + Copy>(
    pub [[BorderType<T>; 2]; N],
);

impl<const N: usize> ConvType<N> {
    pub(crate) fn unfold(self, kernel_size: &[usize; N]) -> ExplicitConv<N> {
        match self {
            ConvType::Full => ExplicitConv {
                pad: kernel_size.map(|kernel| [kernel - 1; 2]),
                stride: std::array::from_fn(|_| 1),
            },
            ConvType::Same => {
                let split = |k_size: usize| {
                    if k_size % 2 == 0 {
                        [(k_size - 1) / 2 + 1, (k_size - 1) / 2]
                    } else {
                        [(k_size - 1) / 2; 2]
                    }
                };

                ExplicitConv {
                    pad: kernel_size.map(split),
                    stride: std::array::from_fn(|_| 1),
                }
            }
            ConvType::Valid => ExplicitConv {
                pad: std::array::from_fn(|_| [0; 2]),
                stride: std::array::from_fn(|_| 1),
            },
            ConvType::Custom(pads, strides) => ExplicitConv {
                pad: pads.map(|pad| [pad; 2]),
                stride: strides,
            },
            ConvType::Explicit(pad, stride) => ExplicitConv { pad, stride },
        }
    }
}

impl<const N: usize, T: num::traits::NumAssign + Copy> PaddingMode<N, T> {
    pub(crate) fn unfold(self) -> ExplictPadding<N, T> {
        match self {
            PaddingMode::Zeros => ExplictPadding([[BorderType::Zeros; 2]; N]),
            PaddingMode::Const(num) => ExplictPadding([[BorderType::Const(num); 2]; N]),
            PaddingMode::Reflect => ExplictPadding([[BorderType::Reflect; 2]; N]),
            PaddingMode::Replicate => ExplictPadding([[BorderType::Replicate; 2]; N]),
            PaddingMode::Circular => ExplictPadding([[BorderType::Circular; 2]; N]),
            PaddingMode::Custom(borders) => ExplictPadding(borders.map(|border| [border; 2])),
            PaddingMode::Explicit(borders) => ExplictPadding(borders),
        }
    }
}

pub trait Padding<const N: usize, T: num::traits::NumAssign + Copy> {
    fn padding(
        &self,
        kernel_size: &[usize; N],
        conv_type: &ConvType<N>,
        padding_mode: &PaddingMode<N, T>,
    ) -> Self;
    fn padding_explicit(
        &self,
        conv_type: &ExplicitConv<N>,
        padding_mode: &ExplictPadding<N, T>,
    ) -> Self;
}

impl<const N: usize, S, T, D> Padding<N, T> for ndarray::ArrayBase<S, D>
where
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
    T: num::traits::NumAssign + Copy,
{
    fn padding(
        &self,
        kernel_size: &[usize; N],
        conv_type: &ConvType<N>,
        padding_mode: &PaddingMode<N, T>,
    ) -> Self {
        self.padding_explicit(&conv_type.unfold(kernel_size), &padding_mode.unfold())
    }

    fn padding_explicit(
        &self,
        conv_type: &ExplicitConv<N>,
        padding_mode: &ExplictPadding<N, T>,
    ) -> Self {
        conv_type.pad.into_iter().zip(padding_mode.0.into_iter());
        todo!()
    }
}

#[derive(Default)]
struct Test<const N: usize, B: Index<usize>, T: num::traits::NumAssign + Copy> {
    strides: PhantomData<B>,
    data: Vec<T>,
}

impl<const N: usize, B: Index<usize>, T: num::traits::NumAssign + Copy>
    Test<N, B, T>
{
    pub fn new() {
        let a = Self {
            strides: PhantomData {},
            data: vec![],
        };

        let strides = a.strides;
    }
}

struct PaddingBuffer<const N: usize, T: num::traits::NumAssign + Copy> {
    strides: [usize; N],
    data: Vec<T>,
}

impl<const N: usize, T: num::traits::NumAssign + Copy> PaddingBuffer<N, T> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<S>(
        data: &ndarray::ArrayBase<S, ndarray::Dim<[usize; N]>>,
        pad_size: &[[usize; 2]; N],
        padding_mode: &PaddingMode<N, T>,
    ) where
        S: ndarray::Data<Elem = T>,
        ndarray::Dim<[usize; N]>: ndarray::Dimension,
    {
        let padding_mode = padding_mode.unfold();

        let mut padding_buf_shape = [0; N];
        padding_buf_shape.copy_from_slice(data.shape());

        padding_buf_shape
            .iter_mut()
            .zip(pad_size.iter())
            .for_each(|(pbs, p)| {
                *pbs += p[0] + p[1];
            });
        let mut pb = ndarray::Array::from_elem(padding_buf_shape.as_slice(), T::zero());

        let mut pb = vec![T::zero(); padding_buf_shape.iter().product()];
    }
    pub fn iter() {}
}

// fn padding_recursion<const N: usize, S, SM, T, D>(
//     src: ndarray::ArrayBase<S, D>,
//     tar: ndarray::ArrayBase<SM, D>,
// ) where
//     S: ndarray::Data<Elem = T>,
//     SM: ndarray::DataMut<Elem = T>,
//     D: ndarray::Dimension + ndarray::RemoveAxis,
//     T: num::traits::NumAssign + Copy,
//     <D as ndarray::Dimension>::Smaller: ndarray::RemoveAxis,
// {
//     match N {
//         1 => {}
//         2.. => {
//             src.axis_iter(Axis(0))
//                 .zip(tar.axis_iter_mut(Axis(0)))
//                 .for_each(|(src, tar)| padding_recursion(src, tar));

//             src;
//         }
//         _ => unreachable!(),
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conv_type_unfold() {
        let a = Test::<2, [usize; 2], f32>::default();
    }

    #[test]
    fn padding_mode_unfold() {}
}


// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn conv_type_unfold() {}

//     #[test]
//     fn padding_mode_unfold() {}

//     #[test]
//     fn axis_iter() {
//         let mut a = ndarray::array![
//             [[1, 2, 3], [5, 6, 7], [8, 9, 10]],
//             [[11, 12, 13], [15, 16, 17], [18, 19, 110]]
//         ];

//         let N = a.raw_dim().ndim() - 1;

//         a.axis_iter_mut(Axis(0)).map(|b| {});

//         for i in 1..N {
//             a.strides()[N - i];
//         }

//         for i in 0..N - 1 {
//             a.axis_iter_mut(Axis(i)).map(|b| {});
//         }

//         dbg!(&a.axis_iter(Axis(0)).collect::<Vec<_>>());

//         dbg!(&a);
//         let b = a.rows().into_iter().collect::<Vec<_>>();
//         dbg!(a.slice_axis(Axis(2), ndarray::Slice::from(..).step_by(2)));
//         dbg!(a.slice(s![..;2, .., ..]));
//         dbg!(b);
//     }
// }
