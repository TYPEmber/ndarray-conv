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

        // let output = Array::uninit(output_shape);
        // let buffer = Array::uninit(output_shape);
        // let mut output = unsafe { output.assume_init() };
        // let mut buffer = unsafe { buffer.assume_init() };

        let mut output = Array::zeros(output_shape);
        let mut buffer = Array::zeros(output_shape);

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
        let raw_dim: [usize; N] = std::array::from_fn(|i| input.raw_dim()[i]);

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
            let fft = self.cp.plan_fft_inverse(buffer.shape()[N - 1]);
            let mut scratch =
                vec![Complex::new(T::zero(), T::zero()); fft.get_outofplace_scratch_len()];

            // contiguous
            buffer = Array::from_shape_vec(buffer.raw_dim(), buffer.into_raw_vec_and_offset().0)
                .unwrap();

            // out-of-place inverse FFT from `input` into `buffer`
            fft.process_outofplace_with_scratch(
                input.as_slice_mut().unwrap(),
                buffer.as_slice_mut().unwrap(),
                &mut scratch,
            );

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
            // no need to check result
            // large input sizes may cause slight numerical issues
            let _ = rp.process_with_scratch(
                input_row.as_slice_mut().unwrap(),
                output_row.as_slice_mut().unwrap(),
                &mut scratch,
            );
        }

        let len = T::from_usize(output.len()).unwrap();
        output.map_mut(|x| *x = x.div(len));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ===== 1D Roundtrip Tests =====

    mod roundtrip_1d {
        use super::*;

        #[test]
        fn basic() {
            let original = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
            let mut input = original.clone();

            let mut p = Processor {
                rp: realfft::RealFftPlanner::new(),
                rp_origin_len: 0,
                cp: rustfft::FftPlanner::new(),
            };

            let mut freq = p.forward(&mut input);
            let reconstructed = p.backward(&mut freq);

            for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "1D Forward->Backward failed. Original: {}, Reconstructed: {}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn different_sizes() {
            // Test various 1D sizes to catch edge cases
            let test_cases = vec![
                array![1.0f64, 2.0],
                array![1.0, 2.0, 3.0],
                array![1.0, 2.0, 3.0, 4.0, 5.0],
            ];

            for (i, original) in test_cases.into_iter().enumerate() {
                let mut input = original.clone();
                let mut p = Processor {
                    rp: realfft::RealFftPlanner::new(),
                    rp_origin_len: 0,
                    cp: rustfft::FftPlanner::new(),
                };

                let mut freq = p.forward(&mut input);
                let reconstructed = p.backward(&mut freq);

                for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                    assert!(
                        (orig - recon).abs() < 1e-10,
                        "1D Test case {} failed. Original: {}, Reconstructed: {}",
                        i,
                        orig,
                        recon
                    );
                }
            }
        }
    }

    // ===== 2D Roundtrip Tests =====

    mod roundtrip_2d {
        use super::*;

        #[test]
        fn basic() {
            let original = array![[1.0f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
            let mut input = original.clone();

            let mut p = Processor {
                rp: realfft::RealFftPlanner::new(),
                rp_origin_len: 0,
                cp: rustfft::FftPlanner::new(),
            };

            let mut freq = p.forward(&mut input);
            let reconstructed = p.backward(&mut freq);

            for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "2D Forward->Backward failed. Original: {}, Reconstructed: {}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn different_sizes() {
            // Test 2x2
            let original = array![[1.0f64, 2.0], [3.0, 4.0]];
            let mut input = original.clone();
            let mut p = Processor {
                rp: realfft::RealFftPlanner::new(),
                rp_origin_len: 0,
                cp: rustfft::FftPlanner::new(),
            };

            let mut freq = p.forward(&mut input);
            let reconstructed = p.backward(&mut freq);

            for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "2D (2x2) test failed. Original: {}, Reconstructed: {}",
                    orig,
                    recon
                );
            }

            // Test 3x3
            let original = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let mut input = original.clone();
            let mut p = Processor {
                rp: realfft::RealFftPlanner::new(),
                rp_origin_len: 0,
                cp: rustfft::FftPlanner::new(),
            };

            let mut freq = p.forward(&mut input);
            let reconstructed = p.backward(&mut freq);

            for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "2D (3x3) test failed. Original: {}, Reconstructed: {}",
                    orig,
                    recon
                );
            }
        }

        #[test]
        fn large_array() {
            use ndarray_rand::rand_distr::Uniform;
            use ndarray_rand::RandomExt;

            // Test large array that might trigger edge cases
            let original = Array::random((200, 5000), Uniform::new(0f32, 1f32).unwrap());
            let mut input = original.clone();

            let mut p = Processor {
                rp: realfft::RealFftPlanner::new(),
                rp_origin_len: 0,
                cp: rustfft::FftPlanner::new(),
            };

            let mut freq = p.forward(&mut input);
            let reconstructed = p.backward(&mut freq);

            // Check a sample of values
            let sample_indices = vec![(0, 0), (0, 100), (100, 0), (100, 2500), (199, 4999)];
            for &(i, j) in &sample_indices {
                let orig = original[[i, j]];
                let recon = reconstructed[[i, j]];
                assert!(
                    (orig - recon).abs() < 1e-6,
                    "Large 2D test failed at ({}, {}). Original: {}, Reconstructed: {}, Diff: {}",
                    i,
                    j,
                    orig,
                    recon,
                    (orig - recon).abs()
                );
            }

            // Check overall statistics
            let max_diff = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (o - r).abs())
                .fold(0.0f32, |acc, x| acc.max(x));

            assert!(
                max_diff < 1e-6,
                "Maximum reconstruction error {} exceeds tolerance",
                max_diff
            );
        }
    }

    // ===== 3D Roundtrip Tests =====

    mod roundtrip_3d {
        use super::*;

        #[test]
        fn basic() {
            let original = array![
                [[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ];
            let mut input = original.clone();

            let mut p = Processor {
                rp: realfft::RealFftPlanner::new(),
                rp_origin_len: 0,
                cp: rustfft::FftPlanner::new(),
            };

            let mut a_fft = p.forward(&mut input);
            let reconstructed = p.backward(&mut a_fft);

            for (orig, recon) in original.iter().zip(reconstructed.iter()) {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "3D Forward->Backward failed. Original: {}, Reconstructed: {}, Diff: {}",
                    orig,
                    recon,
                    (orig - recon).abs()
                );
            }
        }
    }

    // ===== Low-Level FFT API Tests =====

    mod fft_api {
        use super::*;
        use rustfft::num_complex::Complex;

        #[test]
        fn manual_complex_fft_roundtrip() {
            // Test using rustfft directly to verify FFT planner usage
            let mut arr = array![[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],]
                .map(|&v| Complex::new(v as f32, 0.0));
            let mut fft = rustfft::FftPlanner::new();

            // Forward FFT
            let row_forward = fft.plan_fft_forward(arr.shape()[1]);
            for mut row in arr.rows_mut() {
                row_forward.process(row.as_slice_mut().unwrap());
            }

            // Transpose
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

            // Backward FFT
            let row_backward = fft.plan_fft_inverse(arr.shape()[1]);
            for mut row in arr.rows_mut() {
                row_backward.process(row.as_slice_mut().unwrap());
            }

            // Transpose back
            let mut arr = Array::from_shape_vec(
                [arr.shape()[1], arr.shape()[0]],
                arr.permuted_axes([1, 0]).iter().copied().collect(),
            )
            .unwrap();

            let row_backward = fft.plan_fft_inverse(arr.shape()[1]);
            for mut row in arr.rows_mut() {
                row_backward.process(row.as_slice_mut().unwrap());
            }

            // Verify reconstruction (should be close to original [[1,2,3,4], ...])
            for val in arr.iter() {
                let expected_re = val.re.round();
                assert!(
                    (val.re - expected_re).abs() < 1e-5,
                    "FFT roundtrip failed. Got {}, expected approximately {}",
                    val.re,
                    expected_re
                );
                assert!(
                    val.im.abs() < 1e-5,
                    "Imaginary part should be near zero, got {}",
                    val.im
                );
            }
        }
    }
}
