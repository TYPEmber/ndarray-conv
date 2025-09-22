use super::*;
use crate::{dilation::WithDilation, ReverseKernel};
use ndarray::prelude::*;
use num::traits::FromPrimitive;

fn assert_eq_tch<T, const N: usize>(res: Array<T, Dim<[usize; N]>>, res_tch: tch::Tensor)
where
    T: PartialEq + FromPrimitive,
    Dim<[usize; N]>: Dimension,
{
    let tch_res = Array::from_iter(res_tch.reshape(res.len() as i64).iter::<f64>().unwrap())
        .to_shape(res.shape())
        .unwrap()
        .map(|v| T::from_f64(v.round()).unwrap());

    assert!(tch_res.iter().eq(res.iter()));
}

fn get_tch_shape(shape: &[usize]) -> Vec<i64> {
    std::iter::repeat_n(1, 2)
        .chain(shape.iter().map(|v| *v as i64))
        .collect::<Vec<_>>()
}

/// Tests the `conv` function with a 2D array and a kernel.
#[test]
fn tch_conv2d() {
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

    let kernel = tch::Tensor::from_slice(kernel.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape(get_tch_shape(kernel.shape()));

    let result = tensor
        .f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1)
        .unwrap();
    // result.unwrap().print();

    assert_eq_tch(res, result);
}

/// Tests basic convolution operations with various parameters.
#[test]
fn test_conv() {
    let arr = array![[1, 2], [3, 4]];
    let kernel = array![[1, 1], [1, 1]];

    let res = arr
        .conv(
            &kernel,
            ConvMode::Custom {
                padding: [1, 2],
                strides: [2, 2],
            },
            PaddingMode::Zeros,
        )
        .unwrap();
    assert_eq!(res, array![[0, 3, 0], [0, 7, 0]]);
    dbg!(res);

    let arr = array![[1, 2], [3, 4]];
    let kernel = array![[1, 1], [1, 1]];

    let res = arr
        .conv(&kernel, ConvMode::Full, PaddingMode::Zeros)
        .unwrap();
    assert_eq!(res, array![[1, 3, 2], [4, 10, 6], [3, 7, 4]]);
    dbg!(res);

    let arr = array![1, 2, 3, 4, 5, 6];
    let kernel = array![1, 1, 1];

    let res = arr
        .conv(
            &kernel,
            ConvMode::Custom {
                padding: [4],
                strides: [2],
            },
            PaddingMode::Zeros,
        )
        .unwrap();
    assert_eq!(res, array![0, 1, 6, 12, 11, 0]);
    dbg!(res);

    let arr = array![1, 2, 3, 4, 5, 6];
    let kernel = array![1, 1, 1];

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
    assert_eq!(res, array![1, 4, 9, 8, 5]);
    dbg!(res);
}

/// Tests if the convolution result aligns with LibTorch
#[test]
fn aligned_with_libtorch() {
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
    let kernel = tch::Tensor::from_slice(kernel.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape(get_tch_shape(kernel.shape()));

    let res_tch = tensor
        .f_conv1d::<tch::Tensor>(&kernel, None, 2, 4, 2, 1)
        .unwrap();

    assert_eq_tch(res, res_tch);

    //

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

    // reverse kernel in tch
    let kernel = tch::Tensor::from_slice(
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
        .f_conv1d::<tch::Tensor>(&kernel, None, 2, 4, 2, 1)
        .unwrap();

    assert_eq_tch(res, res_tch);

    //

    let arr = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    let kernel = array![[1, 1, 1], [1, 1, 1,], [1, 1, 1]];

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
    let kernel = tch::Tensor::from_slice(kernel.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape(get_tch_shape(kernel.shape()));

    let res_tch = tensor
        .f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1)
        .unwrap();

    assert_eq_tch(res, res_tch);

    //

    let arr = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    let kernel = array![[1, 1, 1], [1, 1, 1]];

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
    let kernel = tch::Tensor::from_slice(kernel.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape(get_tch_shape(kernel.shape()));

    let res_tch = tensor
        .f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1)
        .unwrap();

    assert_eq_tch(res, res_tch);

    //

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
    let kernel = tch::Tensor::from_slice(kernel.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape(get_tch_shape(kernel.shape()));

    let res_tch = tensor
        .f_conv3d::<tch::Tensor>(&kernel, None, [1, 2, 1], 2, 2, 1)
        .unwrap();

    assert_eq_tch(res, res_tch);
}
