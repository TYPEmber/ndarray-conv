use num::complex::ComplexFloat;

/// A simple example and test program demonstrating convolution operations
/// using the `ndarray-conv` crate. It performs both standard and FFT-based
/// convolutions and compares their results for correctness.
fn main() {
    use ndarray::prelude::*;
    use ndarray_conv::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::time::Instant;

    let mut small_duration = 0u128;
    let test_cycles_small = 10;
    // small input images
    for i in 0..100 {
        for _ in 0..test_cycles_small {
            let x = Array::random((1007 + i, 4007 + i), Uniform::new(0f32, 1.));
            let k = Array::random((27, 21), Uniform::new(0f32, 1.));
            // let x = Array::random(20000 + i, Uniform::new(0f32, 1.));
            // let k = Array::random(200, Uniform::new(0f32, 1.));

            let now = Instant::now();
            let a = x
                .conv(k.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
                .unwrap();
            let b = x
                .conv_fft(k.with_dilation(2), ConvMode::Same, PaddingMode::Zeros)
                .unwrap();

            // dbg!(a.shape(), b.shape());

            // dbg!(&a, &b);

            let d = a - b;
            assert!(d.iter().all(|&v| v.abs() < 1e-3));

            // dbg!(d);

            // let mut x = x.permuted_axes([1,0]);
            // let mut buffer = Array::uninit(x.raw_dim());
            // buffer.zip_mut_with(&x, |transpose, &origin| {
            //     transpose.write(origin);
            // });
            // x = unsafe { buffer.assume_init() };

            // let _ =Array::from_shape_vec(
            //     x.raw_dim(),
            //     x.permuted_axes([1,0]).iter().copied().collect(),
            // );

            // naive_conv::conv_2d(&x, &k);
            // x.conv_2d_fft(
            //     &k,
            //     PaddingSize::Same,
            //     PaddingMode::Custom([BorderType::Reflect, BorderType::Circular]),
            // );
            // ndarray_conv::conv_2d::fft::conv_2d::<f64, ndarray::OwnedRepr<f64>, ndarray::OwnedRepr<f64>>(
            //     &x, &k,
            // );
            // ndarray_conv::conv_2d::ndrustfft::conv_2d::<f64, ndarray::OwnedRepr<f64>, ndarray::OwnedRepr<f64>>(
            //     &x, &k,
            // );
            small_duration += now.elapsed().as_nanos();
        }
        println!(
            "Time for {i} arrays, {} iterations: {} milliseconds",
            test_cycles_small,
            small_duration / 1_000_000
        );
        small_duration = 0;
    }
}
