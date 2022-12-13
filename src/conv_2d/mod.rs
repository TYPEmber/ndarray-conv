use ndarray::prelude::*;
use num::traits::NumAssign;

mod fft;

pub trait Conv2DExt<T: NumAssign + Copy> {
    type Output;
    type IxD;

    fn conv_2d<SK: ndarray::Data<Elem = T>>(
        &self,
        kernel: &ArrayBase<SK, Self::IxD>,
    ) -> Option<Self::Output>;
}

impl<T, S> Conv2DExt<T> for ArrayBase<S, Ix2>
where
    S: ndarray::DataMut<Elem = T>,
    T: Copy + Clone + NumAssign + std::fmt::Debug + std::fmt::Display + Send + Sync,
{
    type Output = ndarray::Array<T, Ix2>;
    type IxD = Ix2;

    fn conv_2d<SK: ndarray::Data<Elem = T>>(
        &self,
        kernel: &ArrayBase<SK, Self::IxD>,
    ) -> Option<Self::Output> {
        let padding = (kernel.shape()[0] / 2 * 2, kernel.shape()[1] / 2 * 2);
        let mut buf =
            Array::<T, Ix2>::zeros([self.shape()[0] + padding.0, self.shape()[1] + padding.1]);
        let buf_shape = buf.shape().to_vec();
        let mut sub_buf = buf.slice_mut(s!(
            padding.0 / 2..buf.shape()[0] - padding.0 / 2,
            padding.1 / 2..buf.shape()[1] - padding.1 / 2
        ));
        sub_buf.assign(self);

        // dbg!(&sub_buf);

        let mut ret = Array::<T, Ix2>::zeros(self.dim());

        let mut offset = vec![];
        for r in -(kernel.shape()[1] as isize / 2)..=kernel.shape()[1] as isize / 2 {
            for c in -(kernel.shape()[0] as isize / 2)..=kernel.shape()[0] as isize / 2 {
                offset.push((
                    r * buf_shape[1] as isize + c,
                    kernel[[
                        (r + kernel.shape()[1] as isize / 2) as usize,
                        (c + kernel.shape()[0] as isize / 2) as usize,
                    ]],
                ));
            }
        }

        unsafe {
            if self.len() >= 32 * 32 {
                ndarray::Zip::from(&mut ret)
                    .and(&sub_buf)
                    .par_for_each(|r, s| {
                        let mut temp = T::zero();
                        for (o, k) in offset.iter() {
                            temp += *(s as *const T).offset(*o) * *k
                        }
                        *r = temp;
                    });
            } else {
                ndarray::Zip::from(&mut ret).and(&sub_buf).for_each(|r, s| {
                    let mut temp = T::zero();
                    for (o, k) in offset.iter() {
                        temp += *(s as *const T).offset(*o) * *k
                    }
                    *r = temp;
                });
            }

            // ret.zip_mut_with(&sub_buf, |r, s| {
            //     let mut temp = T::zero();
            //     for (o, k) in offset.iter() {
            //         temp += *(s as *const T).offset(*o) * *k
            //     }
            //     *r = temp;
            // });

            // for (i, item) in sub_buf.iter_mut().enumerate() {
            //     let mut temp = T::zero();
            //     for (o, k) in offset.iter() {
            //         temp += *(item as *mut T).offset(*o) * *k
            //     }
            //     *ret.as_mut_ptr().add(i) = temp;
            // }
        }

        // dbg!(&buf);

        // ndarray::Zip::indexed(buf.windows(kernel.dim())).for_each(|ixd, window| {
        //     let ixd = ixd.into_dimension();
        //     let temp = &window * &kernel;

        //     let ri = ixd[0];
        //     let ci = ixd[1];
        //     unsafe { *ret.uget_mut(ndarray::Ix2(ri, ci)) = temp.sum() };
        // });

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

        assert_ne!(dbg!(input_pixels.conv_2d(&kernel)), None);
        assert_eq!(
            dbg!(input_pixels.conv_2d(&kernel)).unwrap(),
            output_pixels
        );
    }

    #[test]
    fn test_stride() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let a = input_pixels.slice(s!(..,..; 2));

        dbg!(a);
    }
}
