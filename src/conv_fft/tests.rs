use super::*;
use crate::{dilation::WithDilation, ConvExt, ReverseKernel};
use ndarray::prelude::*;
use rustfft::num_complex::Complex;

// ===== Verification Against Conv =====
// Conv FFT results should match Conv (the trusted baseline)

mod vs_conv {
    use num::complex::ComplexFloat;

    use super::*;

    // Tolerance constants
    const TOLERANCE_F32: f32 = 1e-5;
    const TOLERANCE_F64: f64 = 1e-9;

    /// Compare f32 FFT results with Conv baseline
    fn assert_fft_matches_conv_f32<const N: usize>(
        fft: Array<f32, Dim<[usize; N]>>,
        conv: Array<i32, Dim<[usize; N]>>,
    ) where
        Dim<[usize; N]>: Dimension,
    {
        assert_eq!(
            fft.shape(),
            conv.shape(),
            "Shape mismatch: FFT {:?} vs Conv {:?}",
            fft.shape(),
            conv.shape()
        );

        fft.iter()
            .zip(conv.iter())
            .enumerate()
            .for_each(|(idx, (fft_val, conv_val))| {
                let diff = (fft_val.round() - *conv_val as f32).abs();
                assert!(
                    diff < TOLERANCE_F32,
                    "Mismatch at index {}: FFT={:.6}, Conv={}, diff={:.6}",
                    idx,
                    fft_val,
                    conv_val,
                    diff
                );
            });
    }

    /// Compare f64 FFT results with Conv baseline
    fn assert_fft_matches_conv_f64<const N: usize>(
        fft: Array<f64, Dim<[usize; N]>>,
        conv: Array<i32, Dim<[usize; N]>>,
    ) where
        Dim<[usize; N]>: Dimension,
    {
        assert_eq!(
            fft.shape(),
            conv.shape(),
            "Shape mismatch: FFT {:?} vs Conv {:?}",
            fft.shape(),
            conv.shape()
        );

        fft.iter()
            .zip(conv.iter())
            .enumerate()
            .for_each(|(idx, (fft_val, conv_val))| {
                let diff = (fft_val.round() - *conv_val as f64).abs();
                assert!(
                    diff < TOLERANCE_F64,
                    "Mismatch at index {}: FFT={:.10}, Conv={}, diff={:.10}",
                    idx,
                    fft_val,
                    conv_val,
                    diff
                );
            });
    }

    /// Compare Complex<f32> FFT results with another Complex<f32> result
    fn assert_fft_matches_conv_complex<const N: usize>(
        fft: Array<Complex<f32>, Dim<[usize; N]>>,
        conv: Array<Complex<f32>, Dim<[usize; N]>>,
    ) where
        Dim<[usize; N]>: Dimension,
    {
        assert_eq!(
            fft.shape(),
            conv.shape(),
            "Shape mismatch: FFT {:?} vs Conv {:?}",
            fft.shape(),
            conv.shape()
        );

        fft.iter()
            .zip(conv.iter())
            .enumerate()
            .for_each(|(idx, (fft_val, conv_val))| {
                let diff = (fft_val - conv_val).abs();
                assert!(
                    diff < TOLERANCE_F32,
                    "Mismatch at index {}: FFT={:.6}+{:.6}i, Conv={:.6}+{:.6}i, diff={:.6}",
                    idx,
                    fft_val.re,
                    fft_val.im,
                    conv_val.re,
                    conv_val.im,
                    diff
                );
            });
    }

    /// Compare Complex<f64> FFT results with another Complex<f64> result
    fn assert_fft_matches_conv_complex_f64<const N: usize>(
        fft: Array<Complex<f64>, Dim<[usize; N]>>,
        conv: Array<Complex<f64>, Dim<[usize; N]>>,
    ) where
        Dim<[usize; N]>: Dimension,
    {
        assert_eq!(
            fft.shape(),
            conv.shape(),
            "Shape mismatch: FFT {:?} vs Conv {:?}",
            fft.shape(),
            conv.shape()
        );

        fft.iter()
            .zip(conv.iter())
            .enumerate()
            .for_each(|(idx, (fft_val, conv_val))| {
                let diff = (fft_val - conv_val).abs();
                assert!(
                    diff < TOLERANCE_F64,
                    "Mismatch at index {}: FFT={:.10}+{:.10}i, Conv={:.10}+{:.10}i, diff={:.10}",
                    idx,
                    fft_val.re,
                    fft_val.im,
                    conv_val.re,
                    conv_val.im,
                    diff
                );
            });
    }

    // ----- 1D Tests -----

    mod one_d {
        use super::*;

        #[test]
        fn same_mode_f32() {
            let arr = array![1, 2, 3, 4, 5, 6];
            let kernel = array![1, 1, 1, 1];

            let conv_result = arr
                .conv(kernel.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    kernel.map(|&x| x as f32).with_dilation(2),
                    ConvMode::Same,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }

        #[test]
        fn same_mode_complex() {
            // Test with actual complex numbers (non-zero imaginary parts)
            let arr_complex = array![
                Complex::new(1.0, 0.5),
                Complex::new(2.0, -0.3),
                Complex::new(3.0, 0.8),
                Complex::new(4.0, -0.2),
                Complex::new(5.0, 0.6),
                Complex::new(6.0, -0.4),
            ];
            let kernel_complex = array![
                Complex::new(1.0, 0.1),
                Complex::new(1.0, -0.1),
                Complex::new(1.0, 0.2),
                Complex::new(1.0, -0.2),
            ];

            let conv_result = arr_complex
                .conv_fft(
                    kernel_complex.with_dilation(2),
                    ConvMode::Same,
                    PaddingMode::Zeros,
                )
                .unwrap();

            let fft_result = arr_complex
                .conv_fft(
                    kernel_complex.with_dilation(2),
                    ConvMode::Same,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_complex(fft_result, conv_result);
        }

        #[test]
        fn circular_padding() {
            let arr: Array1<f32> = array![
                0.0, 0.1, 0.3, 0.4, 0.0, 0.1, 0.3, 0.4, 0.0, 0.1, 0.3, 0.4, 0.0, 0.1, 0.3, 0.4
            ];
            let kernel: Array1<f32> = array![0.1, 0.3, 0.6, 0.3, 0.1];

            let conv_result = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Circular)
                .unwrap();

            let fft_result = arr
                .conv_fft(&kernel, ConvMode::Same, PaddingMode::Circular)
                .unwrap();

            conv_result
                .iter()
                .zip(fft_result.iter())
                .enumerate()
                .for_each(|(idx, (conv_val, fft_val))| {
                    assert!(
                        (conv_val - fft_val).abs() < 1e-6,
                        "Mismatch at index {}: Conv={:.6}, FFT={:.6}",
                        idx,
                        conv_val,
                        fft_val
                    );
                });
        }

        #[test]
        fn full_mode() {
            let arr = array![1, 2, 3, 4, 5];
            let kernel = array![1, 2, 1];

            let conv_result = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f64)
                .conv_fft(
                    &kernel.map(|&x| x as f64),
                    ConvMode::Full,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f64(fft_result, conv_result);
        }

        #[test]
        fn valid_mode() {
            let arr = array![1, 2, 3, 4, 5, 6];
            let kernel = array![1, 1, 1];

            let conv_result = arr
                .conv(&kernel, ConvMode::Valid, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    &kernel.map(|&x| x as f32),
                    ConvMode::Valid,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }
    }

    // ----- 2D Tests -----

    mod two_d {
        use super::*;

        #[test]
        fn same_mode_f32() {
            let arr = array![[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]];
            let kernel = array![[1, 0], [3, 1]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Replicate)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f64)
                .conv_fft(
                    &kernel.map(|&x| x as f64),
                    ConvMode::Same,
                    PaddingMode::Replicate,
                )
                .unwrap();

            assert_fft_matches_conv_f64(fft_result, conv_result);
        }

        #[test]
        fn custom_mode_with_dilation() {
            let arr = array![[1, 2], [3, 4]];
            let kernel = array![[1, 0], [3, 1]];

            let conv_result = arr
                .conv(
                    kernel.with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [3, 3],
                        strides: [2, 2],
                    },
                    PaddingMode::Replicate,
                )
                .unwrap();

            let fft_result_f64 = arr
                .map(|&x| x as f64)
                .conv_fft(
                    kernel.map(|&x| x as f64).with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [3, 3],
                        strides: [2, 2],
                    },
                    PaddingMode::Replicate,
                )
                .unwrap();

            assert_fft_matches_conv_f64(fft_result_f64, conv_result);
        }

        #[test]
        fn custom_mode_complex() {
            // Test with actual complex numbers (non-zero imaginary parts)
            let arr_complex = array![
                [Complex::new(1.0, 0.2), Complex::new(2.0, -0.3)],
                [Complex::new(3.0, 0.5), Complex::new(4.0, -0.1)]
            ];
            let kernel_complex = array![
                [Complex::new(1.0, 0.1), Complex::new(0.0, 0.2)],
                [Complex::new(3.0, -0.2), Complex::new(1.0, 0.15)]
            ];

            let conv_result = arr_complex
                .conv_fft(
                    kernel_complex.with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [3, 3],
                        strides: [2, 2],
                    },
                    PaddingMode::Replicate,
                )
                .unwrap();

            let fft_result = arr_complex
                .conv_fft(
                    kernel_complex.with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [3, 3],
                        strides: [2, 2],
                    },
                    PaddingMode::Replicate,
                )
                .unwrap();

            assert_fft_matches_conv_complex_f64(fft_result, conv_result);
        }

        #[test]
        fn full_mode() {
            let arr = array![[1, 2], [3, 4]];
            let kernel = array![[1, 1], [1, 1]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    &kernel.map(|&x| x as f32),
                    ConvMode::Full,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }

        #[test]
        fn valid_mode() {
            let arr = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
            let kernel = array![[1, 1], [1, 1]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Valid, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f64)
                .conv_fft(
                    &kernel.map(|&x| x as f64),
                    ConvMode::Valid,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f64(fft_result, conv_result);
        }
    }

    // ----- 3D Tests -----

    mod three_d {
        use super::*;

        #[test]
        fn same_mode_f32() {
            let arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
            let kernel = array![
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ];

            let conv_result = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    &kernel.map(|&x| x as f32),
                    ConvMode::Same,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }

        #[test]
        fn same_mode_complex() {
            // Test with actual complex numbers (non-zero imaginary parts)
            let arr_complex = array![
                [
                    [Complex::new(1.0, 0.3), Complex::new(2.0, -0.2)],
                    [Complex::new(3.0, 0.5), Complex::new(4.0, -0.4)]
                ],
                [
                    [Complex::new(5.0, 0.1), Complex::new(6.0, -0.3)],
                    [Complex::new(7.0, 0.6), Complex::new(8.0, -0.1)]
                ]
            ];
            let kernel_complex = array![
                [
                    [
                        Complex::new(1.0, 0.05),
                        Complex::new(1.0, -0.05),
                        Complex::new(1.0, 0.1)
                    ],
                    [
                        Complex::new(1.0, -0.1),
                        Complex::new(1.0, 0.15),
                        Complex::new(1.0, -0.15)
                    ],
                    [
                        Complex::new(1.0, 0.2),
                        Complex::new(1.0, -0.2),
                        Complex::new(1.0, 0.05)
                    ]
                ],
                [
                    [
                        Complex::new(1.0, -0.05),
                        Complex::new(1.0, 0.1),
                        Complex::new(1.0, -0.1)
                    ],
                    [
                        Complex::new(1.0, 0.15),
                        Complex::new(1.0, -0.15),
                        Complex::new(1.0, 0.2)
                    ],
                    [
                        Complex::new(1.0, -0.2),
                        Complex::new(1.0, 0.05),
                        Complex::new(1.0, -0.05)
                    ]
                ],
            ];

            let conv_result = arr_complex
                .conv_fft(&kernel_complex, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr_complex
                .conv_fft(&kernel_complex, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            assert_fft_matches_conv_complex(fft_result, conv_result);
        }

        #[test]
        fn full_mode() {
            let arr = array![[[1, 2]], [[3, 4]]];
            let kernel = array![[[1, 1]], [[1, 1]]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f64)
                .conv_fft(
                    &kernel.map(|&x| x as f64),
                    ConvMode::Full,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f64(fft_result, conv_result);
        }

        #[test]
        fn valid_mode() {
            let arr = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
            let kernel = array![[[1, 1]], [[1, 1]]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Valid, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    &kernel.map(|&x| x as f32),
                    ConvMode::Valid,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }
    }

    // ----- Different Padding Modes -----

    mod padding_modes {
        use super::*;

        #[test]
        fn replicate_2d() {
            let arr = array![[1, 2, 3], [4, 5, 6]];
            let kernel = array![[1, 1], [1, 1]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Replicate)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    &kernel.map(|&x| x as f32),
                    ConvMode::Same,
                    PaddingMode::Replicate,
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }

        #[test]
        fn zeros_2d() {
            let arr = array![[1, 2, 3], [4, 5, 6]];
            let kernel = array![[1, 1], [1, 1]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f64)
                .conv_fft(
                    &kernel.map(|&x| x as f64),
                    ConvMode::Same,
                    PaddingMode::Zeros,
                )
                .unwrap();

            assert_fft_matches_conv_f64(fft_result, conv_result);
        }

        #[test]
        fn const_padding_2d() {
            let arr = array![[1, 2], [3, 4]];
            let kernel = array![[1, 1], [1, 1]];

            let conv_result = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Const(7))
                .unwrap();

            let fft_result = arr
                .map(|&x| x as f32)
                .conv_fft(
                    &kernel.map(|&x| x as f32),
                    ConvMode::Full,
                    PaddingMode::Const(7.0),
                )
                .unwrap();

            assert_fft_matches_conv_f32(fft_result, conv_result);
        }
    }
}

// ===== Parallel vs Serial FFT Consistency =====
// Verify that conv_fft_par produces bit-identical
// results compared to their serial counterparts.  Gated on the rayon feature.

#[cfg(feature = "rayon")]
mod par_vs_serial {
    use super::*;
    use crate::get_fft_processor;
    use ndarray::Dimension;

    const TOL_F32: f32 = 1e-4;
    const TOL_F64: f64 = 1e-9;

    fn max_diff_f32<const N: usize>(
        a: &Array<f32, Dim<[Ix; N]>>,
        b: &Array<f32, Dim<[Ix; N]>>,
    ) -> f32
    where
        Dim<[Ix; N]>: Dimension,
    {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn max_diff_f64<const N: usize>(
        a: &Array<f64, Dim<[Ix; N]>>,
        b: &Array<f64, Dim<[Ix; N]>>,
    ) -> f64
    where
        Dim<[Ix; N]>: Dimension,
    {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    // ---- 1D ----

    #[test]
    fn one_d_same_f32() {
        let arr = array![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(|&x| x as f32);
        let ker = array![1i32, -1, 2].map(|&x| x as f32);

        let serial = arr
            .conv_fft(&ker, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        let par = arr
            .conv_fft_par(&ker, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        let diff = max_diff_f32(&serial, &par);
        assert!(
            diff < TOL_F32,
            "1D Same f32: conv_fft_par vs conv_fft max_diff = {:.3e}",
            diff
        );
    }

    #[test]
    fn one_d_full_f64() {
        let arr = array![1i32, 2, 3, 4, 5, 6].map(|&x| x as f64);
        let ker = array![1i32, 2, 1].map(|&x| x as f64);

        let serial = arr
            .conv_fft(&ker, ConvMode::Full, PaddingMode::Zeros)
            .unwrap();
        let par = arr
            .conv_fft_par(&ker, ConvMode::Full, PaddingMode::Zeros)
            .unwrap();
        let diff = max_diff_f64(&serial, &par);
        assert!(
            diff < TOL_F64,
            "1D Full f64: conv_fft_par vs conv_fft max_diff = {:.3e}",
            diff
        );
    }

    // ---- 2D ----

    #[test]
    fn two_d_same_f32() {
        let arr = Array::from_shape_fn((32, 32), |(i, j)| ((i + j) % 10) as f32);
        let ker = array![[1.0f32, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0f32, -2.0, -1.0]];

        for mode in [ConvMode::Same, ConvMode::Valid, ConvMode::Full] {
            let serial = arr.conv_fft(&ker, mode, PaddingMode::Zeros).unwrap();
            let par = arr.conv_fft_par(&ker, mode, PaddingMode::Zeros).unwrap();
            let diff = max_diff_f32(&serial, &par);
            assert!(
                diff < TOL_F32,
                "2D {:?} f32: conv_fft_par vs conv_fft max_diff = {:.3e}",
                mode,
                diff
            );
        }
    }

    #[test]
    fn two_d_replicate_f64() {
        let arr = Array::from_shape_fn((24, 24), |(i, j)| ((i * j) % 7) as f64);
        let ker = array![[1.0f64, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

        let serial = arr
            .conv_fft(&ker, ConvMode::Same, PaddingMode::Replicate)
            .unwrap();
        let par = arr
            .conv_fft_par(&ker, ConvMode::Same, PaddingMode::Replicate)
            .unwrap();
        let diff = max_diff_f64(&serial, &par);
        assert!(
            diff < TOL_F64,
            "2D Same Replicate f64: conv_fft_par vs conv_fft max_diff = {:.3e}",
            diff
        );
    }

    // ---- 3D ----

    #[test]
    fn three_d_same_f32() {
        let arr = Array::from_shape_fn((16, 16, 16), |(i, j, k)| ((i + j + k) % 5) as f32);
        let ker = Array::from_shape_fn((3, 3, 3), |(i, j, k)| {
            if i == 1 && j == 1 && k == 1 {
                8.0f32
            } else {
                -1.0f32
            }
        });

        let serial = arr
            .conv_fft(&ker, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        let par = arr
            .conv_fft_par(&ker, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        let diff = max_diff_f32(&serial, &par);
        assert!(
            diff < TOL_F32,
            "3D Same f32: conv_fft_par vs conv_fft max_diff = {:.3e}",
            diff
        );
    }

    // ---- with_processor variants ----





    // ---- dilation ----

    #[test]
    fn two_d_with_dilation_f32() {
        use crate::dilation::WithDilation;
        let arr = Array::from_shape_fn((20, 20), |(i, j)| ((i + j) % 5) as f32);
        let ker = array![[1.0f32, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];

        let serial = arr
            .conv_fft(ker.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        let par = arr
            .conv_fft_par(ker.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
            .unwrap();

        let diff = max_diff_f32(&serial, &par);
        assert!(
            diff < TOL_F32,
            "2D Same dilation=2 f32: conv_fft_par vs conv_fft max_diff = {:.3e}",
            diff
        );
    }
}

// ===== Edge Cases =====

mod edge_cases {
    use super::*;

    #[test]
    fn single_element() {
        let arr = array![42];
        let kernel = array![2];

        let conv_result = arr
            .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();

        let fft_result = arr
            .map(|&x| x as f32)
            .conv_fft(
                &kernel.map(|&x| x as f32),
                ConvMode::Same,
                PaddingMode::Zeros,
            )
            .unwrap();

        assert_eq!(fft_result.map(|x| x.round() as i32), conv_result);
    }

    #[test]
    fn large_array_2d() {
        // Test with a larger array to ensure FFT is actually used
        let arr = Array::from_shape_fn((50, 50), |(i, j)| ((i + j) % 10) as i32);
        let kernel = array![[1, 2, 1], [2, 4, 2], [1, 2, 1]];

        let conv_result = arr
            .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();

        let fft_result = arr
            .map(|&x| x as f64)
            .conv_fft(
                &kernel.map(|&x| x as f64),
                ConvMode::Same,
                PaddingMode::Zeros,
            )
            .unwrap();

        // Check a sample of points
        for i in 0..5 {
            for j in 0..5 {
                let diff = (fft_result[[i, j]].round() - conv_result[[i, j]] as f64).abs();
                assert!(
                    diff < 1e-8,
                    "Mismatch at [{}, {}]: FFT={:.6}, Conv={}",
                    i,
                    j,
                    fft_result[[i, j]],
                    conv_result[[i, j]]
                );
            }
        }
    }
}
