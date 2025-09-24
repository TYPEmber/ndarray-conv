use super::Processor as ProcessorTrait;
use super::*;

/// Complex-valued FFT processor backed by `rustfft`.
///
/// This struct plans complex-to-complex FFTs and provides helper methods
/// that operate on n-dimensional `ndarray::Array` values. The algorithm
/// always transforms along the last axis (which is contiguous) and then
/// permutes axes so the next axis becomes the last. That keeps per-lane
/// operations contiguous and avoids many reallocations.
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
    /// input: mutable reference to **contiguous** input array
    pub fn internal<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
        scratch: &mut [Complex<T>],
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
        let buffer = Array::uninit(input.raw_dim());
        let mut output = unsafe { output.assume_init() };
        let mut buffer = unsafe { buffer.assume_init() };

        let fft = self.cp.plan_fft(output.shape()[N - 1], direction);

        fft.process_outofplace_with_scratch(
            input.as_slice_mut().unwrap(),
            output.as_slice_mut().unwrap(),
            scratch,
        );

        // axes permutation helper: initial axes rotated right by 1. We will
        // `permuted_axes(axes)` before each subsequent stage so the next
        // dimension becomes the last (contiguous) axis.
        let mut axes: [usize; N] = std::array::from_fn(|i| i);

        match direction {
            rustfft::FftDirection::Forward => axes.rotate_right(1),
            rustfft::FftDirection::Inverse => axes.rotate_left(1),
        };

        // perform FFT on last axis, then permute and repeat for remaining axes
        for _ in 0..N - 1 {
            output = output.permuted_axes(axes);

            // reshape
            buffer = Array::from_shape_vec(output.raw_dim(), buffer.into_raw_vec_and_offset().0)
                .unwrap();
            buffer.zip_mut_with(&output, |transpose, &origin| {
                *transpose = origin;
            });

            // continugous
            output = Array::from_shape_vec(output.raw_dim(), output.into_raw_vec_and_offset().0)
                .unwrap();

            let fft = self.cp.plan_fft(output.shape()[N - 1], direction);

            fft.process_outofplace_with_scratch(
                buffer.as_slice_mut().unwrap(),
                output.as_slice_mut().unwrap(),
                scratch,
            );
        }

        output
    }

    pub fn backward_internal<const N: usize>(
        &mut self,
        mut input: Array<Complex<T>, Dim<[Ix; N]>>,
        mut scratch: Option<&mut Vec<Complex<T>>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        // Mirror of `forward_internal`: avoid extra copies by allocating two
        // buffers and swapping them while permuting axes. We perform an
        // inverse FFT along the last axis into `output`, then rotate/permute
        // and continue for each remaining axis. Using two buffers lets us
        // reuse memory and avoid repeated allocations and cloning.
        let output = Array::uninit(input.raw_dim());
        let buffer = Array::uninit(input.raw_dim());
        let mut output = unsafe { output.assume_init() };
        let mut buffer = unsafe { buffer.assume_init() };

        let cp = self.cp.plan_fft_inverse(input.shape()[N - 1]);

        let input_raw_dim = std::array::from_fn(|i| input.raw_dim()[i]);
        for (mut input_row, mut output_row) in input.rows_mut().into_iter().zip(output.rows_mut()) {
            match scratch.as_mut() {
                Some(s) => cp.process_outofplace_with_scratch(
                    input_row.as_slice_mut().unwrap(),
                    output_row.as_slice_mut().unwrap(),
                    s,
                ),
                None => cp.process_outofplace_with_scratch(
                    input_row.as_slice_mut().unwrap(),
                    output_row.as_slice_mut().unwrap(),
                    &mut self.get_scratch::<N>(input_raw_dim),
                ),
            }
        }

        // axes permutation helper: initial axes rotated left by 1. Each loop
        // iteration does: permute -> copy into transpose buffer -> run
        // inverse FFT on the (new) last axis. The copy step is necessary to
        // make lanes contiguous for the in-place FFT routines.
        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_left(1);

        // perform inverse FFT on last axis, then permute and repeat for remaining axes
        for _ in 0..N - 1 {
            output = output.permuted_axes(axes);

            let mut transpose_buffer =
                Array::from_shape_vec(output.raw_dim(), buffer.into_raw_vec_and_offset().0)
                    .unwrap();
            transpose_buffer.zip_mut_with(&output, |transpose, &origin| {
                *transpose = origin;
            });
            buffer = output;
            output = transpose_buffer;

            let cp = self.cp.plan_fft_inverse(output.shape()[N - 1]);
            for mut lane in output.lanes_mut(ndarray::Axis(N - 1)) {
                match scratch.as_mut() {
                    Some(s) => cp.process_with_scratch(lane.as_slice_mut().unwrap(), s),
                    None => cp.process(lane.as_slice_mut().unwrap()),
                }
            }
        }

        let len = Complex::new(T::from_usize(output.len()).unwrap(), T::zero());
        output.map_mut(|x| *x = *x / len);
        output
    }
}

impl<T: FftNum> ProcessorTrait<T, Complex<T>> for Processor<T> {
    // fn get_processor() -> Self {
    //     Self::default()
    // }
    
    #[allow(clippy::uninit_vec)]
    fn get_scratch<const N: usize>(&mut self, input_dim: [usize; N]) -> Vec<Complex<T>> {
        let max_len = input_dim
            .iter()
            .copied()
            .map(|dim| self.cp.plan_fft_forward(dim).get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let mut scratch = Vec::with_capacity(max_len);
        unsafe { scratch.set_len(max_len) };
        scratch
    }

    fn forward<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        let mut scratch = self.get_scratch::<N>(std::array::from_fn(|i| input.raw_dim()[i]));
        self.internal(input, &mut scratch, rustfft::FftDirection::Forward)
    }

    fn backward<const N: usize>(
        &mut self,
        mut input: Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        let mut scratch = self.get_scratch::<N>(std::array::from_fn(|i| input.raw_dim()[i]));

        let mut output = self.internal(&mut input, &mut scratch, rustfft::FftDirection::Inverse);
        let len = Complex::new(T::from_usize(output.len()).unwrap(), T::zero());
        output.map_mut(|x| *x = *x / len);
        output
        // self.backward_internal(input, None)
    }

    fn forward_with_scratch<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
        scratch: &mut Vec<Complex<T>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        self.internal(input, scratch, rustfft::FftDirection::Forward)
    }

    fn backward_with_scratch<const N: usize>(
        &mut self,
        mut input: Array<Complex<T>, Dim<[Ix; N]>>,
        scratch: &mut Vec<Complex<T>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        let mut output = self.internal(&mut input, scratch, rustfft::FftDirection::Inverse);
        let len = Complex::new(T::from_usize(output.len()).unwrap(), T::zero());
        output.map_mut(|x| *x = *x / len);
        output
        // self.backward_internal(input, Some(scratch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_forward_backward_1d() {
        let mut proc = Processor::<f32>::default();
        let arr = array![
            Complex::new(1.0f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0)
        ];
        let freq = proc.forward(&mut arr.clone());
        let recon = proc.backward(freq);
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
        let freq = proc.forward(&mut arr.clone());
        let recon = proc.backward(freq);
        for (a, b) in arr.iter().zip(recon.iter()) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn test_scratch_methods() {
        let mut proc = Processor::<f32>::default();
        let input_dim = [4, 2];
        let mut arr = Array2::<Complex<f32>>::zeros((4, 2));
        let mut scratch = proc.get_scratch(input_dim);
        let freq = proc.forward_with_scratch(&mut arr, &mut scratch);
        let recon = proc.backward_with_scratch(freq, &mut scratch);
        for v in recon.iter() {
            assert!(v.re.abs() < 1e-4 && v.im.abs() < 1e-4);
        }
    }

    // #[test]
    // process_outofplace_with_scratch: 327.333µs
    // process_immutable_with_scratch: 331.25µs
    // process_with_scratch: 368.792µs
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

        let iters = 200usize;

        // Warm up
        for _ in 0..10 {
            fft.process_immutable_with_scratch(
                input.as_mut_slice(),
                out.as_mut_slice(),
                &mut scratch,
            );
            fft.process_with_scratch(inplace.as_mut_slice(), &mut scratch);
        }

        // measure immutable/out-of-place (immutable API)
        let t_immutable = Instant::now();
        for _ in 0..iters {
            fft.process_immutable_with_scratch(
                input.as_mut_slice(),
                out.as_mut_slice(),
                &mut scratch,
            );
        }
        let dur_immutable = t_immutable.elapsed();

        // measure explicit out-of-place with scratch
        let t_outofplace = Instant::now();
        for _ in 0..iters {
            fft.process_outofplace_with_scratch(
                input.as_mut_slice(),
                out.as_mut_slice(),
                &mut scratch,
            );
        }
        let dur_outofplace = t_outofplace.elapsed();

        // measure in-place
        let t_inplace = Instant::now();
        for _ in 0..iters {
            // copy input back into inplace buffer each iteration
            inplace.copy_from_slice(&input);
            fft.process_with_scratch(inplace.as_mut_slice(), &mut scratch);
        }
        let dur_inplace = t_inplace.elapsed();

        println!(
            "process_immutable_with_scratch: {:?}, process_outofplace_with_scratch: {:?}, process_with_scratch: {:?}",
            dur_immutable, dur_outofplace, dur_inplace
        );

        // The test intentionally doesn't assert on speed; it only prints timings.
        // Keep it as a smoke check so it always passes.
    }
}
