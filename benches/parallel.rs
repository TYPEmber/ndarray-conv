use criterion::{criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_conv::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

/// Benchmark comparing conv, conv_fft, conv_fft_par across different data/kernel sizes.
fn bench_parallel(c: &mut Criterion) {
    // ===== 1D benchmarks =====
    {
        let x = Array::random(10000, Uniform::new(0f32, 1.).unwrap());
        let k = Array::random(31, Uniform::new(0f32, 1.).unwrap());

        let mut fft_processor = get_fft_processor::<f32, f32>();

        c.bench_function("1d_conv", |b| {
            b.iter(|| x.conv(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("1d_fft", |b| {
            b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("1d_fft_with_processor", |b| {
            b.iter(|| {
                x.conv_fft_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });

        c.bench_function("1d_fft_par", |b| {
            b.iter(|| x.conv_fft_par(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("1d_fft_par_with_processor", |b| {
            b.iter(|| {
                x.conv_fft_par_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });
    }

    // ===== 2D benchmarks: medium data, small kernel =====
    {
        let x = Array::random((500, 500), Uniform::new(0f32, 1.).unwrap());
        let k = Array::random((7, 7), Uniform::new(0f32, 1.).unwrap());

        let mut fft_processor = get_fft_processor::<f32, f32>();

        c.bench_function("2d_500x500_k7_conv", |b| {
            b.iter(|| x.conv(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_500x500_k7_fft", |b| {
            b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_500x500_k7_fft_proc", |b| {
            b.iter(|| {
                x.conv_fft_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });

        c.bench_function("2d_500x500_k7_fft_par", |b| {
            b.iter(|| x.conv_fft_par(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_500x500_k7_fft_par_proc", |b| {
            b.iter(|| {
                x.conv_fft_par_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });
    }

    // ===== 2D benchmarks: large data, small kernel =====
    {
        let x = Array::random((2049, 2049), Uniform::new(0f32, 1.).unwrap());
        let k = Array::random((7, 7), Uniform::new(0f32, 1.).unwrap());

        let mut fft_processor = get_fft_processor::<f32, f32>();

        c.bench_function("2d_2049x2049_k7_conv", |b| {
            b.iter(|| x.conv(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_2049x2049_k7_fft", |b| {
            b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_2049x2049_k7_fft_proc", |b| {
            b.iter(|| {
                x.conv_fft_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });

        c.bench_function("2d_2049x2049_k7_fft_par", |b| {
            b.iter(|| x.conv_fft_par(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_2049x2049_k7_fft_par_proc", |b| {
            b.iter(|| {
                x.conv_fft_par_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });
    }

    // ===== 2D benchmarks: large data, larger kernel =====
    {
        let x = Array::random((2049, 2049), Uniform::new(0f32, 1.).unwrap());
        let k = Array::random((31, 31), Uniform::new(0f32, 1.).unwrap());

        let mut fft_processor = get_fft_processor::<f32, f32>();

        c.bench_function("2d_2049x2049_k31_conv", |b| {
            b.iter(|| x.conv(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_2049x2049_k31_fft", |b| {
            b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_2049x2049_k31_fft_proc", |b| {
            b.iter(|| {
                x.conv_fft_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });

        c.bench_function("2d_2049x2049_k31_fft_par", |b| {
            b.iter(|| x.conv_fft_par(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_2049x2049_k31_fft_par_proc", |b| {
            b.iter(|| {
                x.conv_fft_par_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });
    }

    // ===== 2D benchmarks: medium data, medium kernel =====
    {
        let x = Array::random((200, 5000), Uniform::new(0f32, 1.).unwrap());
        let k = Array::random((11, 31), Uniform::new(0f32, 1.).unwrap());

        let mut fft_processor = get_fft_processor::<f32, f32>();

        c.bench_function("2d_200x5000_k11x31_fft", |b| {
            b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_200x5000_k11x31_fft_proc", |b| {
            b.iter(|| {
                x.conv_fft_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });

        c.bench_function("2d_200x5000_k11x31_fft_par", |b| {
            b.iter(|| x.conv_fft_par(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("2d_200x5000_k11x31_fft_par_proc", |b| {
            b.iter(|| {
                x.conv_fft_par_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });
    }

    // ===== 3D benchmarks =====
    {
        let x = Array::random((10, 100, 200), Uniform::new(0f32, 1.).unwrap());
        let k = Array::random((5, 11, 31), Uniform::new(0f32, 1.).unwrap());

        let mut fft_processor = get_fft_processor::<f32, f32>();

        c.bench_function("3d_fft", |b| {
            b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("3d_fft_proc", |b| {
            b.iter(|| {
                x.conv_fft_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });

        c.bench_function("3d_fft_par", |b| {
            b.iter(|| x.conv_fft_par(&k, ConvMode::Same, PaddingMode::Zeros))
        });

        c.bench_function("3d_fft_par_proc", |b| {
            b.iter(|| {
                x.conv_fft_par_with_processor(
                    &k,
                    ConvMode::Same,
                    PaddingMode::Zeros,
                    &mut fft_processor,
                )
            })
        });
    }
}

criterion_group!(benches, bench_parallel);
criterion_main!(benches);
