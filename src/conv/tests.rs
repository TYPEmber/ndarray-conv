use super::*;
use crate::{dilation::WithDilation, ReverseKernel};
use ndarray::prelude::*;
use num::traits::FromPrimitive;

// ===== Helper Functions =====

fn assert_eq_tch<T, const N: usize>(res: Array<T, Dim<[usize; N]>>, res_tch: tch::Tensor)
where
    T: PartialEq + FromPrimitive + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
{
    let tch_res = Array::from_iter(res_tch.reshape(res.len() as i64).iter::<f64>().unwrap())
        .to_shape(res.shape())
        .unwrap()
        .map(|v| T::from_f64(v.round()).unwrap())
        .into_dimensionality::<Dim<[usize; N]>>()
        .unwrap();

    assert_eq!(
        res, tch_res,
        "Conv result doesn't match LibTorch.\nGot: {:?}\nExpected: {:?}",
        res, tch_res
    );
}

fn get_tch_shape(shape: &[usize]) -> Vec<i64> {
    std::iter::repeat_n(1, 2)
        .chain(shape.iter().map(|v| *v as i64))
        .collect::<Vec<_>>()
}

// ===== Verification Against LibTorch =====
// These tests establish Conv as the trusted reference implementation

mod vs_torch {
    use super::*;

    // ----- Full Mode -----

    mod full_mode {
        use super::*;

        #[test]
        fn test_1d() {
            let arr = array![1, 2, 3, 4, 5];
            let kernel = array![1, 2, 1];

            let res = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            // Full mode: padding = kernel_size - 1 = 3 - 1 = 2
            let res_tch = tensor
                .f_conv1d::<tch::Tensor>(&kernel_tensor, None, 1, 2, 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn test_2d() {
            let arr = array![[1, 2], [3, 4]];
            let kernel = array![[1, 1], [1, 1]];

            let res = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            // Full mode: padding = kernel_size - 1 = [2 - 1, 2 - 1] = [1, 1]
            let res_tch = tensor
                .f_conv2d::<tch::Tensor>(&kernel_tensor, None, 1, 1, 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn test_3d() {
            let arr = array![[[1, 2]], [[3, 4]]];
            let kernel = array![[[1, 1]], [[1, 1]]];

            let res = arr
                .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            // Full mode: padding = kernel_size - 1 = [2-1, 1-1, 2-1] = [1, 0, 1]
            let res_tch = tensor
                .f_conv3d::<tch::Tensor>(&kernel_tensor, None, 1, [1, 0, 1], 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }
    }

    // ----- Same Mode -----

    mod same_mode {
        use super::*;

        #[test]
        fn test_1d() {
            let arr = array![1, 2, 3, 4, 5];
            let kernel = array![1, 2, 1];

            let res = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "same", 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn test_2d() {
            let arr = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
            let kernel = array![[1, 0, -1], [2, 0, -2], [1, 0, -1]];

            // Default behavior: kernel is reversed
            let res = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));

            // PyTorch doesn't reverse kernel, so we need to reverse it first
            let kernel_reversed = kernel
                .as_slice()
                .unwrap()
                .iter()
                .copied()
                .rev()
                .collect::<Vec<_>>();
            let kernel_tensor = tch::Tensor::from_slice(&kernel_reversed)
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv2d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "same", 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn test_3d() {
            // Use a simpler 3D array with symmetric kernel
            let arr = array![[[1, 2, 3]], [[4, 5, 6]]];
            let kernel = array![[[1, 1, 1]]];

            let res = arr
                .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));

            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            // Same mode with kernel [1, 1, 3]: padding = [0, 0, 1]
            let res_tch = tensor
                .f_conv3d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "same", 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }
    }

    // ----- Valid Mode -----

    mod valid_mode {
        use super::*;

        #[test]
        fn test_1d() {
            let arr = array![1, 2, 3, 4, 5];
            let kernel = array![1, 2, 1];

            let res = arr
                .conv(&kernel, ConvMode::Valid, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "valid", 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn test_2d() {
            let arr = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
            let kernel = array![[1, 1], [1, 1]];

            let res = arr
                .conv(&kernel, ConvMode::Valid, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv2d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "valid", 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn test_3d() {
            let arr = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]];
            let kernel = array![[[1, 1]], [[1, 1]]];

            let res = arr
                .conv(&kernel, ConvMode::Valid, PaddingMode::Zeros)
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv3d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "valid", 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }
    }

    // ----- With Stride -----

    mod with_stride {
        use super::*;

        #[test]
        fn stride_2_1d() {
            let arr = array![1, 2, 3, 4, 5, 6];
            let kernel = array![1, 1, 1];

            let res = arr
                .conv(
                    &kernel,
                    ConvMode::Custom {
                        padding: [1],
                        strides: [2],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d::<tch::Tensor>(&kernel_tensor, None, 2, 1, 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn stride_2_2d() {
            let arr = array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
            let kernel = array![[1, 1], [1, 1]];

            let res = arr
                .conv(
                    &kernel,
                    ConvMode::Custom {
                        padding: [1, 1],
                        strides: [2, 2],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv2d::<tch::Tensor>(&kernel_tensor, None, [2, 2], 1, 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn stride_3_1d() {
            let arr = array![1, 2, 3, 4, 5, 6, 7, 8, 9];
            let kernel = array![1, 2, 1];

            let res = arr
                .conv(
                    &kernel,
                    ConvMode::Custom {
                        padding: [2],
                        strides: [3],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d::<tch::Tensor>(&kernel_tensor, None, 3, 2, 1, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }
    }

    // ----- With Dilation -----

    mod with_dilation {
        use super::*;

        #[test]
        fn dilation_2_1d() {
            let arr = array![1, 2, 3, 4, 5, 6];
            let kernel = array![1, 1, 2];

            let res = arr
                .conv(
                    kernel.with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [4],
                        strides: [2],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d::<tch::Tensor>(&kernel_tensor, None, 2, 4, 2, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn dilation_2_2d() {
            let arr = array![[1, 1, 1], [1, 1, 1], [1, 1, 2]];
            let kernel = array![[2, 1, 1], [1, 1, 1]];

            let res = arr
                .conv(
                    kernel.with_dilation(2).no_reverse(),
                    ConvMode::Same,
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv2d_padding::<tch::Tensor>(&kernel_tensor, None, 1, "same", 2, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn dilation_2_3d() {
            let arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
            let kernel = array![
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ];

            let res = arr
                .conv(
                    kernel.with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [2, 2, 2],
                        strides: [1, 2, 1],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv3d::<tch::Tensor>(&kernel_tensor, None, [1, 2, 1], 2, 2, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }
    }

    // ----- Kernel Reversal -----

    mod kernel_reverse {
        use super::*;

        #[test]
        fn with_reverse() {
            let arr = array![1, 2, 3, 4, 5, 6];
            let kernel = array![1, 1, 2];

            let res = arr
                .conv(
                    kernel.with_dilation(2),
                    ConvMode::Custom {
                        padding: [4],
                        strides: [2],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));

            // Reverse kernel for torch (it expects non-reversed)
            let kernel_tensor = tch::Tensor::from_slice(
                &kernel
                    .as_slice()
                    .unwrap()
                    .iter()
                    .copied()
                    .rev()
                    .collect::<Vec<_>>(),
            )
            .to_dtype(tch::Kind::Float, false, true)
            .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d::<tch::Tensor>(&kernel_tensor, None, 2, 4, 2, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }

        #[test]
        fn no_reverse() {
            let arr = array![1, 2, 3, 4, 5, 6];
            let kernel = array![1, 1, 2];

            let res = arr
                .conv(
                    kernel.with_dilation(2).no_reverse(),
                    ConvMode::Custom {
                        padding: [4],
                        strides: [2],
                    },
                    PaddingMode::Zeros,
                )
                .unwrap();

            let tensor = tch::Tensor::from_slice(arr.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(arr.shape()));
            let kernel_tensor = tch::Tensor::from_slice(kernel.as_slice().unwrap())
                .to_dtype(tch::Kind::Float, false, true)
                .reshape(get_tch_shape(kernel.shape()));

            let res_tch = tensor
                .f_conv1d::<tch::Tensor>(&kernel_tensor, None, 2, 4, 2, 1)
                .unwrap();

            assert_eq_tch(res, res_tch);
        }
    }
}

// ===== Edge Cases =====
// Quick regression tests without external dependencies

mod edge_cases {
    use super::*;

    #[test]
    fn single_element_array() {
        let arr = array![42];
        let kernel = array![2];
        let res = arr
            .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        assert_eq!(res, array![84]);
    }

    #[test]
    fn single_element_kernel() {
        let arr = array![1, 2, 3, 4];
        let kernel = array![3];
        let res = arr
            .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        assert_eq!(res, array![3, 6, 9, 12]);
    }

    #[test]
    fn identity_kernel() {
        let arr = array![[1, 2], [3, 4]];
        let kernel = array![[1]];
        let res = arr
            .conv(&kernel, ConvMode::Same, PaddingMode::Zeros)
            .unwrap();
        assert_eq!(res, arr);
    }
}
