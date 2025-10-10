use ndarray::ArrayViewMut;

use super::Processor as ProcessorTrait;
use super::*;

/// Complex-valued FFT processor backed by `rustfft`.
///
/// Plans complex-to-complex FFTs and provides helpers that operate on
/// n-dimensional `ndarray::Array` values. The implementation always
/// performs FFTs along the last axis (which is kept contiguous), then
/// permutes axes so the next axis becomes the last. This keeps the heavy
/// FFT work on contiguous memory and avoids many small allocations.
pub struct Processor<T: FftNum> {
    cp: rustfft::FftPlanner<T>,
    _phantom: PhantomData<Complex<T>>,
}

impl<T: FftNum> Default for Processor<T> {
    fn default() -> Self {
        Self {
            cp: rustfft::FftPlanner::new(),
            _phantom: Default::default(),
        }
    }
}

impl<T: FftNum> Processor<T> {
    /// Perform an N-D complex FFT along all axes.
    ///
    /// The function expects an array whose last axis is contiguous. It
    /// performs an out-of-place FFT on that axis into an uninitialized
    /// `output` buffer, then rotates axes and repeats so each axis becomes
    /// the last once. `direction` controls forward vs inverse transform.
    pub fn internal<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
        direction: rustfft::FftDirection,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        // Ensure we always run FFTs along the last axis (which is contiguous),
        // then permute the array so the next axis becomes the last. This
        // avoids creating many small slices and keeps the heavy FFT work on
        // contiguous memory.
        let output = Array::uninit(input.raw_dim());
        let mut output = unsafe { output.assume_init() };

        let fft = self.cp.plan_fft(output.shape()[N - 1], direction);

        // prepare scratch space for out-of-place processing
        let mut scratch = vec![Complex::new(T::zero(), T::zero());
            fft.get_outofplace_scratch_len()];

        fft.process_outofplace_with_scratch(
            input.as_slice_mut().unwrap(),
            output.as_slice_mut().unwrap(),
            &mut scratch,
        );

        let mut buffer = input.view_mut();

    // axes permutation helper: rotate so the next dimension becomes the
    // last (contiguous) axis before each subsequent stage.
        let mut axes: [usize; N] = std::array::from_fn(|i| i);

        match direction {
            rustfft::FftDirection::Forward => axes.rotate_right(1),
            rustfft::FftDirection::Inverse => axes.rotate_left(1),
        };

        // perform FFT on last axis, then permute and repeat for remaining axes
        for _ in 0..N - 1 {
            output = output.permuted_axes(axes);

            // reshape
            buffer = unsafe { ArrayViewMut::from_shape_ptr(output.raw_dim(), buffer.as_mut_ptr()) };
            buffer.zip_mut_with(&output, |transpose, &origin| {
                *transpose = origin;
            });

            // continugous
            output = Array::from_shape_vec(output.raw_dim(), output.into_raw_vec_and_offset().0)
                .unwrap();

            let fft = self.cp.plan_fft(output.shape()[N - 1], direction);
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
}

impl<T: FftNum> ProcessorTrait<T, Complex<T>> for Processor<T> {
    fn forward<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        self.internal(input, rustfft::FftDirection::Forward)
    }

    fn backward<const N: usize>(
        &mut self,
        input: &mut Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        let mut output = self.internal(input, rustfft::FftDirection::Inverse);
        let len = Complex::new(T::from_usize(output.len()).unwrap(), T::zero());
        output.map_mut(|x| *x = *x / len);
        output
        // self.backward_internal(input, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_forward_backward_1d() {
        let mut proc = Processor::<f32>::default();
        let arr = array![
            Complex::new(1.0f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0)
        ];
        let mut freq = proc.forward(&mut arr.clone());
        let recon = proc.backward(&mut freq);
        for (a, b) in arr.iter().zip(recon.iter()) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn test_forward_backward_2d() {
        let mut proc = Processor::<f32>::default();
        let arr = array![
            [Complex::new(1.0f32, 0.0), Complex::new(2.0, 0.0)],
            [Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]
        ];
        let mut freq = proc.forward(&mut arr.clone());
        let recon = proc.backward(&mut freq);
        for (a, b) in arr.iter().zip(recon.iter()) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    /// process_outofplace_with_scratch: 327.333µs
    /// process_immutable_with_scratch: 331.25µs
    /// process_with_scratch: 368.792µs
    fn bench_process_methods_timing() {
        // Compare process_immutable_with_scratch (out-of-place) vs process_with_scratch (in-place)
        // on a single large 1-D FFT to get measurable timings. We print durations
        // so the developer can run this test with `--nocapture` and examine results.
        use std::time::Instant;

        let mut planner = rustfft::FftPlanner::<f32>::new();
        let n = 1024usize; // FFT size
        let fft = planner.plan_fft_forward(n);

        // prepare buffers
        let mut input = vec![Complex::new(1.0f32, 0.0); n];
        let mut out = vec![Complex::new(0.0f32, 0.0); n];
        let mut inplace = input.clone();

        // scratch sized from fft plan
        let scratch_len = fft.get_inplace_scratch_len();
        let mut scratch = vec![Complex::new(0.0f32, 0.0); scratch_len];

        let iters = 700usize;

        // Warm up
        for _ in 0..1000 {
            fft.process_immutable_with_scratch(
                input.as_mut_slice(),
                out.as_mut_slice(),
                &mut scratch,
            );
            fft.process_with_scratch(inplace.as_mut_slice(), &mut scratch);
        }

        // let t_get_scratch = Instant::now();
        // for _ in 0..iters {
        //     let mut a = vec![Complex::<f32>::zero(); fft.get_outofplace_scratch_len()];
        //     fft.process_outofplace_with_scratch(input.as_mut_slice(), out.as_mut_slice(), &mut a);
        // }
        // let dur_get_scratch = t_get_scratch.elapsed();

        // dbg!(dur_get_scratch);

        // measure explicit out-of-place with scratch
        // let t_outofplace = Instant::now();
        // for _ in 0..iters {
        //     fft.process_outofplace_with_scratch(
        //         input.as_mut_slice(),
        //         out.as_mut_slice(),
        //         &mut scratch,
        //     );
        // }
        // let dur_outofplace = t_outofplace.elapsed();

        // dbg!(dur_outofplace);

        // measure immutable/out-of-place (immutable API)
        // let t_immutable = Instant::now();
        // for _ in 0..iters {
        //     fft.process_immutable_with_scratch(
        //         input.as_mut_slice(),
        //         out.as_mut_slice(),
        //         &mut scratch,
        //     );
        // }
        // let dur_immutable = t_immutable.elapsed();

        // dbg!(dur_immutable);

        // measure in-place
        let t_inplace = Instant::now();
        for _ in 0..iters {
            // copy input back into inplace buffer each iteration
            inplace.copy_from_slice(&input);
            fft.process_with_scratch(inplace.as_mut_slice(), &mut scratch);
        }
        let dur_inplace = t_inplace.elapsed();

        dbg!(dur_inplace);

        // println!(
        //     "process_immutable_with_scratch: {:?}, process_outofplace_with_scratch: {:?}, process_with_scratch: {:?}, process_get_scratch: {:?}",
        //     dur_immutable, dur_outofplace, dur_inplace, dur_get_scratch
        // );

        // The test intentionally doesn't assert on speed; it only prints timings.
        // Keep it as a smoke check so it always passes.
    }
}
