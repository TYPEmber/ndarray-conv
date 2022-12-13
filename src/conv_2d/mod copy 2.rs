use std::{fmt::Display, ops::Add};

use ndarray::{prelude::*, IntoDimension};
use num::traits::NumAssign;

pub trait Conv2DExt<T: NumAssign + Copy> {
    type Output;
    type IxD;

    fn conv_2d<SK: ndarray::Data<Elem = T>>(
        &self,
        kernel: ArrayBase<SK, Self::IxD>,
    ) -> Option<Self::Output>;
}

impl<T, S> Conv2DExt<T> for ArrayBase<S, Ix2>
where
    S: ndarray::DataMut<Elem = T>,
    T: Copy + Clone + NumAssign + std::fmt::Debug + Display,
{
    type Output = ndarray::Array<T, Ix2>;
    type IxD = Ix2;

    fn conv_2d<SK: ndarray::Data<Elem = T>>(
        &self,
        kernel: ArrayBase<SK, Self::IxD>,
    ) -> Option<Self::Output> {
        let padding = (kernel.shape()[0] / 2 * 2, kernel.shape()[1] / 2 * 2);
        let mut buf =
            Array::<T, Ix2>::zeros([self.shape()[0] + padding.0, self.shape()[1] + padding.1]);
        buf.slice_mut(s!(
            padding.0 / 2..buf.shape()[0] - padding.0 / 2,
            padding.1 / 2..buf.shape()[1] - padding.1 / 2
        )).assign(self);
        // dbg!(&buf);
        let mut ret = Array::<T, Ix2>::zeros(self.dim());

        ndarray::Zip::indexed(buf.windows(kernel.dim())).for_each(|ixd, window| {
            let ixd = ixd.into_dimension();
            let temp = &window * &kernel;

            let ri = ixd[0];
            let ci = ixd[1];
            unsafe { *ret.uget_mut(ndarray::Ix2(ri, ci)) = temp.sum() };
        });

        Some(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test() {
        // let input_pixels = array![
        //     [
        //         [1, 1, 1, 0, 0],
        //         [0, 1, 1, 1, 0],
        //         [0, 0, 1, 1, 1],
        //         [0, 0, 1, 1, 0],
        //         [0, 1, 1, 0, 0],
        //     ],
        //     [
        //         [1, 1, 1, 0, 0],
        //         [0, 1, 1, 1, 0],
        //         [0, 0, 1, 1, 1],
        //         [0, 0, 1, 1, 0],
        //         [0, 1, 1, 0, 0],
        //     ],
        //     // [
        //     //     [1, 1, 1, 0, 0],
        //     //     [0, 1, 1, 1, 0],
        //     //     [0, 0, 1, 1, 1],
        //     //     [0, 0, 1, 1, 0],
        //     //     [0, 1, 1, 0, 0],
        //     // ]
        // ];

        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
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

        // let kernel = array![
        //     [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        //     [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        //     // [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        // ];
        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        assert_ne!(dbg!(input_pixels.conv_2d(kernel.view())), None);
        assert_eq!(dbg!(input_pixels.conv_2d(kernel.view())).unwrap(), output_pixels);
    }
}
