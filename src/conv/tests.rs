use super::*;
use crate::dilation::WithDilation;
use ndarray::prelude::*;

#[test]
fn tch_conv2d() {
    // let a = vec![1, 2, 3];
    // let b = a
    //     .iter()
    //     .flat_map(|&v| std::iter::repeat(v).take(4))
    //     .collect::<Vec<_>>();
    let tensor = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 3, 3]);
    let kernel = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1]])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 2, 3]);
    // let kernel = tch::Tensor::from_slice2(&[[1]])
    //     .to_dtype(tch::Kind::Float, false, true)
    //     .reshape([1, 1, 1, 1]);
    // let result = tensor.f_conv2d::<tch::Tensor>(&kernel, None, 1, [1, 1], 3i64, 1);
    let result = tensor.f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1);
    result.unwrap().print();

    let arr = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    let kernel = array![[1, 1, 1], [1, 1, 1]];

    let res = arr
        .conv(kernel.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
        .unwrap();
    dbg!(res);
}

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

#[test]
fn aligned_with_libtorch() {
    let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 6]);
    let kernel = tch::Tensor::from_slice(&[1, 1, 1])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 3]);

    let result = tensor.f_conv1d::<tch::Tensor>(&kernel, None, 2, 4, 2, 1);
    result.unwrap().print();

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

    //

    let tensor = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 3, 3]);
    let kernel = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 3, 3]);

    let result = tensor.f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1);
    result.unwrap().print();

    let arr = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    let kernel = array![[1, 1, 1], [1, 1, 1,], [1, 1, 1]];

    let res = arr
        .conv(kernel.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
        .unwrap();
    assert_eq!(res, array![[4, 2, 4], [2, 1, 2], [4, 2, 4]]);
    dbg!(res);

    //

    let tensor = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 3, 3]);
    let kernel = tch::Tensor::from_slice2(&[[1, 1, 1], [1, 1, 1]])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 2, 3]);

    let result = tensor.f_conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 2, 1);
    result.unwrap().print();

    let arr = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    let kernel = array![[1, 1, 1], [1, 1, 1]];

    let res = arr
        .conv(kernel.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
        .unwrap();
    assert_eq!(res, array![[2, 1, 2], [4, 2, 4], [2, 1, 2]]);
    dbg!(res);

    //

    let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8])
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 2, 2, 2]);
    let kernel = tch::Tensor::from_slice(&[
        // 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])
    .to_dtype(tch::Kind::Float, false, true)
    .reshape([1, 1, 2, 3, 3]);

    let result = tensor.f_conv3d::<tch::Tensor>(&kernel, None, [1, 2, 1], 2, 2, 1);
    result.unwrap().print();

    let arr = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    let kernel = array![
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ];

    let res = arr
        .conv(
            kernel.with_dilation(2),
            ConvMode::Custom {
                padding: [2, 2, 2],
                strides: [1, 2, 1],
            },
            PaddingMode::Zeros,
        )
        .unwrap();
    assert_eq!(res, array![[[1, 2]], [[5, 6]], [[1, 2]], [[5, 6]]]);
    dbg!(res);
}
