use std::{fmt::Display, ops::Add};

use ndarray::{prelude::*, IntoDimension};
use num::traits::NumAssign;

pub trait Conv2DExt<T: NumAssign + Copy, IxD: ndarray::Dimension> {
    type Output;

    fn conv_2d<SK: ndarray::Data<Elem = T>>(
        &self,
        kernel: ArrayBase<SK, IxD>,
    ) -> Option<Self::Output>;
}

// impl<T, S, Ix2> Conv2DExt<T, Ix2> for ArrayBase<S, Ix2>
// where
//     S: ndarray::DataMut<Elem = T>,
//     T: Copy + Clone + NumAssign + std::fmt::Debug + Display,
// {        type Output = ndarray::Array<T, Ix2>;

//     }


impl<T, S, IxD> Conv2DExt<T, IxD> for ArrayBase<S, IxD>
where
    S: ndarray::DataMut<Elem = T>,
    T: Copy + Clone + NumAssign + std::fmt::Debug + Display,
    IxD: ndarray::Dimension + Copy + ndarray::RemoveAxis,
    ndarray::Dim<[usize; 2]>: ndarray::NdIndex<<IxD as ndarray::Dimension>::Smaller>,
    <IxD as ndarray::Dimension>::Smaller: std::marker::Copy,
{
    type Output = ndarray::Array<T, IxD>;

    fn conv_2d<SK: ndarray::Data<Elem = T>>(
        &self,
        kernel: ArrayBase<SK, IxD>,
    ) -> Option<ndarray::Array<T, IxD>> {
        if self.shape().len() < 2 {
            return None;
        }
        if self
            .shape()
            .iter()
            .zip(kernel.shape().iter())
            .rev()
            .skip(2)
            .fold(false, |flag, (&a, &b)| flag | (a != b))
        {
            return None;
        }

        // let ret = Array::<T, IxD>::uninit(self.dim());
        let mut ret = Array::<T, IxD>::zeros(self.dim());

        if self.shape().len() == 2 {
            ndarray::Zip::indexed(self.windows(kernel.dim())).for_each(|ixd, window| {
                let ixd = ixd.into_dimension();
                let temp = &window * &kernel;

                let k_s = kernel.dim().into_dimension();
                let ri = ixd[0];
                let ci = ixd[1];
                unsafe {
                    *ret.uget_mut(ndarray::Ix2(ri + k_s[0] / 2, ci + k_s[1] / 2)) = temp.sum()
                };
            });
        } else {
            let axis = self.shape().len() - 3;

            ret.axis_iter_mut(Axis(axis))
                .zip(self.axis_iter(Axis(axis)))
                .zip(kernel.axis_iter(Axis(axis)))
                .for_each(|((mut r, s), k)| unsafe {
                    ndarray::Zip::indexed(s.windows(k.dim())).for_each(|ixd, window| {
                        let ixd = ixd.into_dimension();
                        let temp = &window * &k;

                        let k_s = kernel.dim().into_dimension();
                        let ri = ixd[0];
                        let ci = ixd[1];
                        *r.uget_mut(ndarray::Ix2(ri + k_s[0] / 2, ci + k_s[1] / 2)) = temp.sum();
                    });
                    // dbg!(r);
                });
        }

        dbg!(self.dim());

        // Some(unsafe { ret.assume_init() })
        Some(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test() {
        let input_pixels = array![
            [
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0],
            ],
            [
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0],
            ],
            // [
            //     [1, 1, 1, 0, 0],
            //     [0, 1, 1, 1, 0],
            //     [0, 0, 1, 1, 1],
            //     [0, 0, 1, 1, 0],
            //     [0, 1, 1, 0, 0],
            // ]
        ];

        let output_pixels = array![
            [2, 2, 3, 1, 1],
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [0, 2, 2, 1, 1],
        ];

        for a in input_pixels.axis_iter(Axis(0)) {
            dbg!(a);
        }

        let kernel = array![
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            // [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        ];

        assert_ne!(dbg!(input_pixels.conv_2d(kernel.view())), None);
    }
}
