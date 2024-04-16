use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_conv::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

fn criterion_benchmark(c: &mut Criterion) {
    let x = Array::random((1300, 4000), Uniform::new(0f32, 1.));
    let k = Array::random((21, 41), Uniform::new(0f32, 1.));

    c.bench_function("fft", |b| {
        b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
    });

    c.bench_function("fft_with_processor", |b| {
        b.iter(|| x.conv_fft(&k, ConvMode::Same, PaddingMode::Zeros))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
