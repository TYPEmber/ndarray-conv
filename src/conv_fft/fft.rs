use ndarray::{Array, ArrayBase, DataMut, Dim, IntoDimension, Ix, RemoveAxis};
use num::{Complex, Integer};
use rustfft::FftNum;

pub struct Processor<T: FftNum> {
    rp: realfft::RealFftPlanner<T>,
    rp_origin_len: usize,
    // N dims - 1
    // unstable feature needed
    // cps: [rustfft::FftPlanner<T>; N - 1],
    cp: rustfft::FftPlanner<T>,
}

impl<T: FftNum> Default for Processor<T> {
    fn default() -> Self {
        Self {
            rp: Default::default(),
            rp_origin_len: Default::default(),
            cp: rustfft::FftPlanner::new()
        }
    }
}

impl<T: FftNum> Processor<T> {
    pub fn forward<S: DataMut<Elem = T>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        let raw_dim: [usize; N] = std::array::from_fn(|i| input.raw_dim()[i]);

        let rp = self.rp.plan_fft_forward(raw_dim[N - 1]);
        self.rp_origin_len = rp.len();

        let mut output_shape = raw_dim;
        output_shape[N - 1] = rp.complex_len();
        let mut output = Array::zeros(output_shape);

        for (mut input, mut output) in input.rows_mut().into_iter().zip(output.rows_mut()) {
            rp.process(
                input.as_slice_mut().unwrap(),
                output.as_slice_mut().unwrap(),
            )
            .unwrap();
        }

        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_right(1);
        for _ in 0..N - 1 {
            output_shape.rotate_right(1);
            output = Array::from_shape_vec(
                output_shape.into_dimension(),
                output.permuted_axes(axes).iter().copied().collect(),
            )
            .unwrap();

            let cp = self.cp.plan_fft_forward(output_shape[N - 1]);

            cp.process(output.as_slice_mut().unwrap());
        }

        output
    }

    pub fn backward<const N: usize>(
        &mut self,
        mut input: Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<T, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        // at this time, the raw_dim has been routate_left by N - 1 times
        let mut raw_dim: [usize; N] = std::array::from_fn(|i| input.raw_dim()[i]);

        let rp = self.rp.plan_fft_inverse(self.rp_origin_len);

        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_left(1);
        for _ in 0..N - 1 {
            let cp = self.cp.plan_fft_inverse(raw_dim[N - 1]);
            cp.process(input.as_slice_mut().unwrap());

            raw_dim.rotate_left(1);
            input = Array::from_shape_vec(
                raw_dim.into_dimension(),
                input.permuted_axes(axes).iter().copied().collect(),
            )
            .unwrap();
        }

        let mut output_shape = input.raw_dim();
        output_shape[N - 1] = self.rp_origin_len;
        let mut output = Array::zeros(output_shape);

        for (mut input, mut output) in input.rows_mut().into_iter().zip(output.rows_mut()) {
            if input.len().is_odd() {
                unsafe { input.uget_mut(0).im = T::zero() };
                unsafe { input.uget_mut(input.len() - 1).im = T::zero() }
            };

            rp.process(
                input.as_slice_mut().unwrap(),
                output.as_slice_mut().unwrap(),
            )
            .unwrap();
        }

        let len = T::from_usize(output.len()).unwrap();
        output.map_mut(|x| *x = x.div(len));
        output
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis, Dimension, IntoDimension};

    use crate::{conv_fft, ConvMode, PaddingMode};

    use super::*;

    #[test]
    fn index_axis() {
        let a = array![[1, 2, 3], [4, 5, 6]];

        let shape = a.shape();
        for dim in 0..shape.len() {
            for i in 0..shape[dim] {
                dbg!(a.index_axis(Axis(dim), i));
            }
        }
    }

    #[test]
    fn transpose() {
        let a = array![
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ];
        let mut raw_dim = *unsafe {
            (&mut a.raw_dim() as *mut _ as *mut [usize; 3])
                .as_mut()
                .unwrap()
        };
        // dbg!(&a);
        // dbg!(a.t());
        // dbg!(a.t().t());

        let mut axes = [0, 1, 2];

        axes.rotate_right(1);
        raw_dim.rotate_right(1);
        let a = Array::from_shape_vec(raw_dim, a.permuted_axes(axes).iter().copied().collect())
            .unwrap();
        dbg!(&a);

        // axes.rotate_left(1);
        raw_dim.rotate_right(1);
        let a = Array::from_shape_vec(raw_dim, a.permuted_axes(axes).iter().copied().collect())
            .unwrap();
        dbg!(&a);

        // axes.rotate_left(1);
        raw_dim.rotate_right(1);
        let a = Array::from_shape_vec(raw_dim, a.permuted_axes(axes).iter().copied().collect())
            .unwrap();
        dbg!(&a);
    }

    #[test]
    fn test_forward_backward() {
        let mut a = array![
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]]
        ];
        // let mut a = array![1., 2., 3.];
        // let kernel = array![
        //     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        //     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        // ];

        // conv_fft::padding::data(
        //     &a,
        //     PaddingMode::Zeros,
        //     ConvMode::Same.unfold(&kernel),
        //     [2, 2, 3],
        // );

        let mut p = Processor {
            rp: realfft::RealFftPlanner::new(),
            rp_origin_len: 0,
            cp: rustfft::FftPlanner::new(),
        };

        let a_fft = p.forward(&mut a);

        dbg!(&a_fft);

        let a = p.backward(a_fft);

        dbg!(&a);
    }
}
