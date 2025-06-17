use criterion::{criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_conv::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_vision::processing::ConvolutionExt;

/// Benchmark for 1D convolution using `conv_fft` with various libraries.
fn criterion_benchmark(c: &mut Criterion) {
    let x = Array::random(5000, Uniform::new(0f32, 1.));
    let k = Array::random(31, Uniform::new(0f32, 1.));

    let x_crs = x.to_shape((1, 1, 5000)).unwrap().to_owned();
    let k_crs = k.to_shape((1, 1, 1, 31)).unwrap().to_owned();

    let tensor = tch::Tensor::from_slice(x.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 5000]);
    let kernel = tch::Tensor::from_slice(k.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 31]);

    // for (a, b) in x
    //     .conv_fft(&k, ConvMode::Same, PaddingMode::Zeros)
    //     .unwrap()
    //     .iter()
    //     .zip(
    //         tensor
    //             .conv1d_padding::<tch::Tensor>(&kernel, None, 1, "same", 1, 1)
    //             .reshape(5000)
    //             .iter::<f64>()
    //             .unwrap(),
    //     )
    // {
    //     // need to div kernel size
    //     assert!((*a as f64 - b).abs() < 1e-5);
    // }

    let mut fft_processor = FftProcessor::default();

    /// Benchmark for 1D convolution using `conv_fft`.
    c.bench_function("fft_1d", |b| {
        b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
    });

    /// Benchmark for 1D convolution using `conv_fft_with_processor`.
    c.bench_function("fft_with_processor_1d", |b| {
        b.iter(|| {
            x.conv_fft_with_processor(&k, ConvMode::Same, PaddingMode::Zeros, &mut fft_processor)
        })
    });

    c.bench_function("torch_1d", |b| {
        b.iter(|| tensor.conv1d_padding::<tch::Tensor>(&kernel, None, 1, "same", 1, 1))
    });

    // c.bench_function("convolution_rs_1d", |b| {
    //     b.iter(|| {
    //         convolutions_rs::convolutions::ConvolutionLayer::new_tf(
    //             k_crs.clone(),
    //             None,
    //             1,
    //             convolutions_rs::Padding::Same,
    //         )
    //         .convolve(&x_crs)
    //     });
    // });

    // c.bench_function("fftconvolve_1d", |b| {
    //     b.iter(|| fftconvolve::fftconvolve(&x, &k, fftconvolve::Mode::Same))
    // });

    let x = Array::random((200, 5000), Uniform::new(0f32, 1.));
    let k = Array::random((11, 31), Uniform::new(0f32, 1.));

    let x_crs = x.to_shape((1, 200, 5000)).unwrap().to_owned();
    let k_crs = k.to_shape((1, 1, 11, 31)).unwrap().to_owned();

    let x_nvs = x.to_shape((200, 5000, 1)).unwrap().to_owned();
    let k_nvs = k.to_shape((11, 31, 1)).unwrap().to_owned();

    let tensor = tch::Tensor::from_slice(x.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 200, 5000]);
    let kernel = tch::Tensor::from_slice(k.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 11, 31]);

    let mut fft_processor = FftProcessor::default();

    /// Benchmark for 2D convolution using `conv_fft`.
    c.bench_function("fft_2d", |b| {
        b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
    });

    /// Benchmark for 2D convolution using `conv_fft_with_processor`.
    c.bench_function("fft_with_processor_2d", |b| {
        b.iter(|| {
            x.conv_fft_with_processor(&k, ConvMode::Same, PaddingMode::Zeros, &mut fft_processor)
        })
    });

    c.bench_function("torch_2d", |b| {
        b.iter(|| tensor.conv2d_padding::<tch::Tensor>(&kernel, None, 1, "same", 1, 1))
    });

    // c.bench_function("ndarray_vision_2d", |b| {
    //     b.iter(|| x_nvs.conv2d_with_padding(k_nvs.clone(), &ndarray_vision::core::ZeroPadding))
    // });

    // c.bench_function("convolution_rs_2d", |b| {
    //     b.iter(|| {
    //         convolutions_rs::convolutions::ConvolutionLayer::new_tf(
    //             k_crs.clone(),
    //             None,
    //             1,
    //             convolutions_rs::Padding::Same,
    //         )
    //         .convolve(&x_crs)
    //     });
    // });

    // c.bench_function("fftconvolve_2d", |b| {
    //     b.iter(|| fftconvolve::fftconvolve(&x, &k, fftconvolve::Mode::Same))
    // });

    let x = Array::random((10, 100, 200), Uniform::new(0f32, 1.));
    let k = Array::random((5, 11, 31), Uniform::new(0f32, 1.));

    let x_crs = x.to_shape((10, 100, 200)).unwrap().to_owned();
    let k_crs = k.to_shape((1, 5, 11, 31)).unwrap().to_owned();

    let tensor = tch::Tensor::from_slice(x.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 10, 100, 200]);
    let kernel = tch::Tensor::from_slice(k.as_slice().unwrap())
        .to_dtype(tch::Kind::Float, false, true)
        .reshape([1, 1, 5, 11, 31]);

    let mut fft_processor = FftProcessor::default();

    /// Benchmark for 3D convolution using `conv_fft`.
    c.bench_function("fft_3d", |b| {
        b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
    });

    /// Benchmark for 3D convolution using `conv_fft_with_processor`.
    c.bench_function("fft_with_processor_3d", |b| {
        b.iter(|| {
            x.conv_fft_with_processor(&k, ConvMode::Same, PaddingMode::Zeros, &mut fft_processor)
        })
    });

    c.bench_function("torch_3d", |b| {
        b.iter(|| tensor.conv3d_padding::<tch::Tensor>(&kernel, None, 1, "same", 1, 1))
    });

    // c.bench_function("convolution_rs_3d", |b| {
    //     b.iter(|| {
    //         convolutions_rs::convolutions::ConvolutionLayer::new_tf(
    //             k_crs.clone(),
    //             None,
    //             1,
    //             convolutions_rs::Padding::Same,
    //         )
    //         .convolve(&x_crs)
    //     });
    // });

    // c.bench_function("fftconvolve_3d", |b| {
    //     b.iter(|| fftconvolve::fftconvolve(&x, &k, fftconvolve::Mode::Same))
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
