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

        let mut buffer = input.view_mut();

        // axes permutation helper: rotate so the next dimension becomes the
        // last (contiguous) axis before each subsequent stage.
        let mut axes: [usize; N] = std::array::from_fn(|i| i);

        match direction {
            rustfft::FftDirection::Forward => axes.rotate_right(1),
            rustfft::FftDirection::Inverse => axes.rotate_left(1),
        };

        // perform FFT on last axis, then permute and repeat for remaining axes
        for i in 0..N {
            let fft = self.cp.plan_fft(output.shape()[N - 1], direction);
            let mut scratch =
                vec![Complex::new(T::zero(), T::zero()); fft.get_outofplace_scratch_len()];

            fft.process_outofplace_with_scratch(
                buffer.as_slice_mut().unwrap(),
                output.as_slice_mut().unwrap(),
                &mut scratch,
            );

            // permute axes so the next axis becomes the last (contiguous)
            if i != N - 1 {
                output = output.permuted_axes(axes);

                // reshape
                buffer =
                    unsafe { ArrayViewMut::from_shape_ptr(output.raw_dim(), buffer.as_mut_ptr()) };
                buffer.zip_mut_with(&output, |transpose, &origin| {
                    *transpose = origin;
                });

                // continuous
                output =
                    Array::from_shape_vec(output.raw_dim(), output.into_raw_vec_and_offset().0)
                        .unwrap();
            }
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

    // ===== Basic Roundtrip Tests =====

    mod roundtrip {
        use super::*;

        #[test]
        fn test_1d() {
            let mut proc = Processor::<f32>::default();
            let original = array![
                Complex::new(1.0f32, 0.5),
                Complex::new(2.0, -0.25),
                Complex::new(3.0, 1.25),
                Complex::new(4.0, -0.75)
            ];
            let mut input = original.clone();
            let mut freq = proc.forward(&mut input);
            let recon = proc.backward(&mut freq);

            for (orig, recon) in original.iter().zip(recon.iter()) {
                assert!(
                    (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                    "1D roundtrip failed. Original: {:?}, Reconstructed: {:?}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn test_2d() {
            let mut proc = Processor::<f32>::default();
            let original = array![
                [Complex::new(1.0f32, 0.5), Complex::new(2.0, -1.0)],
                [Complex::new(3.0, 1.5), Complex::new(4.0, -0.5)]
            ];
            let mut input = original.clone();
            let mut freq = proc.forward(&mut input);
            let recon = proc.backward(&mut freq);

            for (orig, recon) in original.iter().zip(recon.iter()) {
                assert!(
                    (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                    "2D roundtrip failed. Original: {:?}, Reconstructed: {:?}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn test_3d() {
            let mut proc = Processor::<f32>::default();
            let original = array![
                [
                    [Complex::new(1.0f32, 0.125), Complex::new(2.0, -0.25)],
                    [Complex::new(3.0, 0.375), Complex::new(4.0, -0.5)]
                ],
                [
                    [Complex::new(5.0, 0.625), Complex::new(6.0, -0.75)],
                    [Complex::new(7.0, 0.875), Complex::new(8.0, -1.0)]
                ]
            ];
            let mut input = original.clone();
            let mut freq = proc.forward(&mut input);
            let recon = proc.backward(&mut freq);

            for (orig, recon) in original.iter().zip(recon.iter()) {
                assert!(
                    (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                    "3D roundtrip failed. Original: {:?}, Reconstructed: {:?}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn different_sizes() {
            let test_cases = vec![
                array![Complex::new(1.0f32, 0.5), Complex::new(2.0, -0.25)],
                array![
                    Complex::new(1.0f32, 0.75),
                    Complex::new(2.0, 1.0),
                    Complex::new(3.0, -1.0)
                ],
                array![
                    Complex::new(1.0f32, 0.25),
                    Complex::new(2.0, -0.5),
                    Complex::new(3.0, 0.75),
                    Complex::new(4.0, -1.0),
                    Complex::new(5.0, 1.25)
                ],
            ];

            for (i, original) in test_cases.into_iter().enumerate() {
                let mut proc = Processor::<f32>::default();
                let mut input = original.clone();
                let mut freq = proc.forward(&mut input);
                let recon = proc.backward(&mut freq);

                for (orig, recon) in original.iter().zip(recon.iter()) {
                    assert!(
                        (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                        "Size test case {} failed. Original: {:?}, Reconstructed: {:?}",
                        i,
                        orig,
                        recon
                    );
                }
            }
        }
    }

    // ===== Complex Value Tests =====

    mod complex_values {
        use super::*;

        #[test]
        fn large_imaginary_parts() {
            let mut proc = Processor::<f32>::default();
            let original = array![
                Complex::new(1.0f32, 3.0),
                Complex::new(2.0, -2.5),
                Complex::new(0.5, 4.0),
                Complex::new(-1.0, 2.0)
            ];
            let mut input = original.clone();
            let mut freq = proc.forward(&mut input);
            let recon = proc.backward(&mut freq);

            for (orig, recon) in original.iter().zip(recon.iter()) {
                assert!(
                    (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                    "Large imaginary parts roundtrip failed. Original: {:?}, Reconstructed: {:?}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn pure_imaginary() {
            let mut proc = Processor::<f32>::default();
            // Test with pure imaginary numbers (re = 0, im != 0)
            let original = array![
                Complex::new(0.0f32, 1.0),
                Complex::new(0.0, 2.0),
                Complex::new(0.0, -1.5),
                Complex::new(0.0, 3.0)
            ];
            let mut input = original.clone();
            let mut freq = proc.forward(&mut input);
            let recon = proc.backward(&mut freq);

            for (orig, recon) in original.iter().zip(recon.iter()) {
                assert!(
                    (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                    "Pure imaginary roundtrip failed. Original: {:?}, Reconstructed: {:?}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn mixed_signs() {
            let mut proc = Processor::<f32>::default();
            // Test with various combinations of positive/negative real and imaginary parts
            let original = array![
                [Complex::new(1.0f32, 2.0), Complex::new(-1.0, 2.0)],
                [Complex::new(1.0, -2.0), Complex::new(-1.0, -2.0)]
            ];
            let mut input = original.clone();
            let mut freq = proc.forward(&mut input);
            let recon = proc.backward(&mut freq);

            for (orig, recon) in original.iter().zip(recon.iter()) {
                assert!(
                    (orig.re - recon.re).abs() < 1e-6 && (orig.im - recon.im).abs() < 1e-6,
                    "Mixed signs roundtrip failed. Original: {:?}, Reconstructed: {:?}",
                    orig,
                    recon
                );
            }
        }
    }
}
