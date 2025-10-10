use ndarray::ArrayViewMut;

use super::*;

use super::Processor as ProcessorTrait;

pub struct Processor<T: ConvFftNum> {
    rp: realfft::RealFftPlanner<T>,
    rp_origin_len: usize,
    cp: rustfft::FftPlanner<T>,
}

impl<T: ConvFftNum> Default for Processor<T> {
    fn default() -> Self {
        Self {
            rp: Default::default(),
            rp_origin_len: Default::default(),
            cp: rustfft::FftPlanner::new(),
        }
    }
}

impl<T: ConvFftNum> ProcessorTrait<T, T> for Processor<T> {
    /// Performs a forward FFT on the given input array.
    ///
    /// This computes a real-to-complex FFT on the last axis (contiguous),
    /// producing a complex-valued array where the last axis length is
    /// `rp.complex_len()`. Remaining axes are transformed with complex
    /// FFTs. All scratch buffers are allocated locally and reused where
    /// possible to avoid extra allocations.
    fn forward<S: DataMut<Elem = T>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
    // Do real->complex on the last (contiguous) axis into an
    // uninitialized `output` buffer, then permute and run complex-to-
    // complex FFTs on the remaining axes while swapping two buffers
    // to avoid repeated allocations and copies.
        let raw_dim: [usize; N] = std::array::from_fn(|i| input.raw_dim()[i]);
        let rp = self.rp.plan_fft_forward(raw_dim[N - 1]);
        self.rp_origin_len = rp.len();

        let mut output_shape = raw_dim;
        output_shape[N - 1] = rp.complex_len();

        let output = Array::uninit(output_shape);
        let buffer = Array::uninit(output_shape);
        let mut output = unsafe { output.assume_init() };
        let mut buffer = unsafe { buffer.assume_init() };

        let mut scratch = vec![Complex::new(T::zero(), T::zero()); rp.get_scratch_len()];
        for (mut input_row, mut output_row) in input.rows_mut().into_iter().zip(output.rows_mut()) {
            rp.process_with_scratch(
                input_row.as_slice_mut().unwrap(),
                output_row.as_slice_mut().unwrap(),
                &mut scratch,
            )
            .unwrap();
        }

        // axes permutation helper: rotate right so we make the next axis the
        // last (contiguous) axis on each iteration.
        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_right(1);

        // perform FFT on last axis, then permute and repeat for remaining axes
        for _ in 0..N - 1 {
            output = output.permuted_axes(axes);

            // reshape
            buffer = Array::from_shape_vec(output.raw_dim(), buffer.into_raw_vec_and_offset().0)
                .unwrap();
            buffer.zip_mut_with(&output, |transpose, &origin| {
                *transpose = origin;
            });

            // contiguous
            output = Array::from_shape_vec(output.raw_dim(), output.into_raw_vec_and_offset().0)
                .unwrap();

            let fft = self
                .cp
                .plan_fft(output.shape()[N - 1], rustfft::FftDirection::Forward);
            let mut scratch =
                vec![Complex::new(T::zero(), T::zero()); fft.get_outofplace_scratch_len()];

            fft.process_outofplace_with_scratch(
                buffer.as_slice_mut().unwrap(),
                output.as_slice_mut().unwrap(),
                &mut scratch,
            );
        }

        output
    }

    /// Performs an inverse FFT on the given input array.
    ///
    /// This performs inverse complex-to-complex FFTs on the axes other
    /// than the last, then finishes with a complex-to-real inverse on the
    /// last axis (turning complex frequency data back into real samples).
    /// Like `forward`, scratch buffers are local and reused when possible.
    fn backward<const N: usize>(
        &mut self,
        input: &mut Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<T, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        // Reverse the forward flow: perform inverse complex FFTs on the last
        // axis (for each remaining axis), permuting and copying into a
        // temporary buffer to maintain contiguous layout, then finally run
        // the complex->real inverse on the last axis.
        let mut raw_dim: [usize; N] = std::array::from_fn(|i| input.raw_dim()[i]);

        // one temporary buffer used per iteration; allocated to raw_dim and
        // re-shaped as necessary to reuse its allocation.
        let buffer = Array::uninit(raw_dim);
        let mut buffer = unsafe { buffer.assume_init() };

        // axes permutation helper: rotate left to undo the right rotations
        // performed by forward.
        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_left(1);

        // work on a mutable view of the input so we can copy into it
        let mut input = input.view_mut();

        for _ in 0..N - 1 {
            let fft = self.cp.plan_fft_inverse(raw_dim[N - 1]);
            let mut scratch = vec![Complex::new(T::zero(), T::zero());
                fft.get_outofplace_scratch_len()];

            // reshape `buffer` to match the current (possibly permuted) shape
            buffer = Array::from_shape_vec(input.raw_dim(), buffer.into_raw_vec_and_offset().0)
                .unwrap();

            // out-of-place inverse FFT from `input` into `buffer`
            fft.process_outofplace_with_scratch(
                input.as_slice_mut().unwrap(),
                buffer.as_slice_mut().unwrap(),
                &mut scratch,
            );

            raw_dim.rotate_left(1);

            // permute `buffer` so the next axis becomes the last, then copy
            // its contents back into `input` (which is arranged to be
            // contiguous for the next iteration).
            buffer = buffer.permuted_axes(axes);
            input = unsafe { ArrayViewMut::from_shape_ptr(buffer.raw_dim(), input.as_mut_ptr()) };
            input.zip_mut_with(&buffer, |dst, &src| *dst = src);
        }

        // now inverse real FFT on the last axis
        let rp = self.rp.plan_fft_inverse(self.rp_origin_len);

        let mut output_shape = input.raw_dim();
        output_shape[N - 1] = self.rp_origin_len;
        let mut output = Array::zeros(output_shape);

        let mut scratch = vec![Complex::new(T::zero(), T::zero()); rp.get_scratch_len()];
        for (mut input_row, mut output_row) in input.rows_mut().into_iter().zip(output.rows_mut()) {
            rp.process_with_scratch(
                input_row.as_slice_mut().unwrap(),
                output_row.as_slice_mut().unwrap(),
                &mut scratch,
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
    use super::*;
    use ndarray::{array, Axis};

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

        let mut a_fft = p.forward(&mut a);

        dbg!(&a_fft);

        let a = p.backward(&mut a_fft);

        dbg!(&a);
    }

    #[test]
    fn test_forward_backward_complex() {
        let mut arr = array![[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],]
            .map(|&v| Complex::new(v as f32, 0.0));
        let mut fft = rustfft::FftPlanner::new();

        // forward
        let row_forward = fft.plan_fft_forward(arr.shape()[1]);
        for mut row in arr.rows_mut() {
            row_forward.process(row.as_slice_mut().unwrap());
        }

        // transpose
        let mut arr = Array::from_shape_vec(
            [arr.shape()[1], arr.shape()[0]],
            arr.permuted_axes([1, 0]).iter().copied().collect(),
        )
        .unwrap();

        let row_forward = fft.plan_fft_forward(arr.shape()[1]);
        for mut row in arr.rows_mut() {
            row_forward.process(row.as_slice_mut().unwrap());
        }

        arr /= Complex::new(16.0, 0.0);

        // backward
        let row_backward = fft.plan_fft_inverse(arr.shape()[1]);
        for mut row in arr.rows_mut() {
            row_backward.process(row.as_slice_mut().unwrap());
        }

        // transpose
        let mut arr = Array::from_shape_vec(
            [arr.shape()[1], arr.shape()[0]],
            arr.permuted_axes([1, 0]).iter().copied().collect(),
        )
        .unwrap();

        let row_backward = fft.plan_fft_inverse(arr.shape()[1]);
        for mut row in arr.rows_mut() {
            row_backward.process(row.as_slice_mut().unwrap());
        }

        dbg!(arr);
    }
}
